"""The only SQL in Dispatch.

Every read and write goes through :class:`JobRepository`. Concentrating them here buys
three things that matter more than the indirection costs:

* **State transitions are atomic and validated in one place.** A conditional ``UPDATE``
  that names the expected current state makes an illegal or racing transition impossible,
  rather than merely unlikely.
* **Derived indexes cannot drift.** The FTS row and the ``job_metadata`` index are
  rebuilt by the same methods that write the canonical data, so there is no path that
  updates one without the other.
* **The daemon is the sole writer**, so there is no locking strategy to design. The
  repository assumes this and takes ``BEGIN IMMEDIATE`` for multi-table writes.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from dispatch.core.clock import Clock, SystemClock
from dispatch.core.errors import IllegalTransition, JobNotFound, ValidationError
from dispatch.core.metadata import CaseMetadata, MetadataSpec
from dispatch.core.models import (
    Job,
    JobEvent,
    JobSpec,
    Note,
    Page,
    Sample,
    Sweep,
    SweepSpec,
    new_job_id,
)
from dispatch.core.provenance import Provenance
from dispatch.core.query import SearchQuery
from dispatch.core.states import TERMINAL_STATES, ExitReason, JobState, can_transition
from dispatch.core.tags import normalise_tags
from dispatch.db.connection import transaction
from dispatch.db.rows import (
    event_from_row,
    job_from_row,
    note_from_row,
    provenance_from_row,
    provenance_to_params,
    sample_from_row,
)

__all__ = ["JobRepository"]

DEFAULT_LIMIT: Final = 200
MAX_LIMIT: Final = 1000
"""Hard ceiling on any page. Bounds IPC message size (§7.5) as much as it bounds memory."""

_JOB_COLUMNS: Final = """
    id, seq, name, workdir, solver, solver_binary, cores, ram_estimate_mb, priority,
    state, created_at, started_at, finished_at, exit_code, exit_reason, exit_signal,
    exit_detail, stdout_path, stderr_path, log_path, pid, pid_start_time, metadata,
    runtime_s, peak_rss_mb, mean_cpu_pct, depends_on_job_id, resource_kind, gpus,
    sweep_id, sweep_position, resume_requested, boot_time
"""

# Columns a transition is permitted to set. An allowlist rather than "whatever the caller
# passed" so that a typo'd keyword is an error instead of a silently ignored update.
_TRANSITION_FIELDS: Final = frozenset(
    {
        "started_at",
        "finished_at",
        "exit_code",
        "exit_reason",
        "exit_signal",
        "exit_detail",
        "pid",
        "pid_start_time",
        "runtime_s",
        "peak_rss_mb",
        "mean_cpu_pct",
        "solver_binary",
        # Set when reboot recovery returns a job to the queue, and cleared in the same
        # statement that starts it again -- so the flag cannot outlive the restart it asked
        # for and cause a second one.
        "resume_requested",
        "boot_time",
    }
)


class JobRepository:
    """Persistent storage for jobs and everything attached to them.

    Args:
        conn: An open, migrated connection from :func:`dispatch.db.connection.connect`.
        clock: Time source, injected so tests are deterministic.
        specs: Metadata specs by adapter name, used to decide whether a metadata value is
            indexed as a number or as text. Optional -- without it, types are inferred
            from the stored Python values, which is the right fallback for history whose
            adapter is no longer installed.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        clock: Clock | None = None,
        specs: Mapping[str, MetadataSpec] | None = None,
    ) -> None:
        self._conn = conn
        self._clock = clock or SystemClock()
        self._specs = dict(specs or {})

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection. For diagnostics and tests."""
        return self._conn

    def register_spec(self, spec: MetadataSpec) -> None:
        """Make an adapter's metadata spec known, for typed metadata indexing."""
        self._specs[spec.ref.adapter] = spec

    # -- creation ---------------------------------------------------------------------

    def create(
        self,
        spec: JobSpec,
        *,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        log_path: Path | None = None,
        state: JobState = JobState.QUEUED,
    ) -> Job:
        """Insert a new job and everything attached to it.

        Writes to five tables (``jobs``, ``job_tags``, ``tags``, ``job_metadata``,
        ``jobs_fts``, plus an event row), all inside one immediate transaction, so a job
        is never half-created.

        Args:
            spec: What the user asked for.
            stdout_path: Where the solver's stdout will go. Assigned by the daemon, which
                owns the log layout; the repository only records it.
            stderr_path: As above, for stderr.
            log_path: The working-directory log, when the case directory can hold one.
            state: Initial state. ``QUEUED`` normally; ``REJECTED`` when validation failed
                and the job is being recorded rather than run.

        Returns:
            The created job.
        """
        if state not in (JobState.QUEUED, JobState.HELD, JobState.REJECTED):
            raise ValidationError(f"A job cannot be created in state {state}")

        now = self._clock.now()
        job_id = new_job_id()
        metadata = spec.metadata or CaseMetadata.empty(spec.solver)

        with transaction(self._conn):
            if spec.depends_on_job_id is not None and not self._exists(spec.depends_on_job_id):
                # Checked here as well as by the foreign key so the user gets Dispatch's own
                # error rather than an IntegrityError from sqlite3.
                raise JobNotFound(spec.depends_on_job_id)
            seq = int(
                self._conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM jobs").fetchone()[0]
            )
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, seq, name, workdir, solver, solver_binary, cores, ram_estimate_mb,
                    priority, state, created_at, stdout_path, stderr_path, log_path,
                    metadata, depends_on_job_id, resource_kind, gpus,
                    sweep_id, sweep_position
                ) VALUES (
                    :id, :seq, :name, :workdir, :solver, :solver_binary, :cores, :ram,
                    :priority, :state, :created_at, :stdout, :stderr, :log,
                    :metadata, :depends_on, :resource_kind, :gpus,
                    :sweep_id, :sweep_position
                )
                """,
                {
                    "id": job_id,
                    "seq": seq,
                    "name": spec.name,
                    "workdir": str(spec.workdir),
                    "solver": spec.solver,
                    "solver_binary": spec.solver_binary,
                    "cores": spec.resources.cores,
                    "ram": spec.resources.ram_mb,
                    "priority": spec.priority,
                    "state": state.value,
                    "created_at": now,
                    "stdout": str(stdout_path) if stdout_path else None,
                    "stderr": str(stderr_path) if stderr_path else None,
                    "log": str(log_path) if log_path else None,
                    "resource_kind": spec.resources.kind.value,
                    "gpus": spec.resources.gpus,
                    "metadata": _dumps(metadata.to_json()),
                    "depends_on": spec.depends_on_job_id,
                    "sweep_id": spec.sweep_id,
                    "sweep_position": spec.sweep_position,
                },
            )
            self._write_tags(job_id, spec.tags)
            self._write_metadata_index(job_id, metadata)
            if spec.note:
                self._insert_note(job_id, spec.note, now)
            self._append_event(job_id, "state", f"created in {state.value}", now)
            self._reindex(job_id)

        return self.get(job_id)

    # -- reads ------------------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        """Fetch one job by id.

        Raises:
            JobNotFound: If no such job exists.
        """
        job = self.get_optional(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def get_optional(self, job_id: str) -> Job | None:
        """Fetch one job by id, or ``None``."""
        row = self._conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return job_from_row(row, tags=self._tags_for(job_id))

    def resolve_id(self, prefix: str) -> str:
        """Expand a unique job-id prefix to the full id.

        Users type the first few characters of a UUID; requiring all 36 would make the CLI
        unusable. An ambiguous prefix is an error rather than a guess.

        Raises:
            JobNotFound: If nothing matches.
            ValidationError: If more than one job matches.
        """
        rows = self._conn.execute(
            "SELECT id FROM jobs WHERE id LIKE ? || '%' LIMIT 5", (prefix.lower(),)
        ).fetchall()
        if not rows:
            raise JobNotFound(prefix)
        if len(rows) > 1:
            raise ValidationError(
                f"Job id {prefix!r} is ambiguous: {len(rows)} jobs match. Use more characters.",
                detail={"matches": [r["id"] for r in rows]},
            )
        return str(rows[0]["id"])

    def queued(self) -> Sequence[Job]:
        """Queued jobs in scheduling order: priority descending, then submission order.

        This is the scheduler's hot query, served by a partial index covering only queued
        rows -- so its cost is proportional to the queue, not to history.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_JOB_COLUMNS} FROM jobs
            WHERE state = 'QUEUED'
            ORDER BY priority DESC, seq ASC
            """
        ).fetchall()
        return self._hydrate(rows)

    def active(self) -> Sequence[Job]:
        """Jobs that currently hold a resource allocation (PREPARING or RUNNING)."""
        rows = self._conn.execute(
            f"""
            SELECT {_JOB_COLUMNS} FROM jobs
            WHERE state IN ('PREPARING', 'RUNNING')
            ORDER BY started_at ASC NULLS LAST, seq ASC
            """
        ).fetchall()
        return self._hydrate(rows)

    def recent(self, limit: int = 10) -> Sequence[Job]:
        """Most recently finished jobs, newest first."""
        rows = self._conn.execute(
            f"""
            SELECT {_JOB_COLUMNS} FROM jobs
            WHERE finished_at IS NOT NULL
            ORDER BY finished_at DESC
            LIMIT ?
            """,
            (_clamp(limit),),
        ).fetchall()
        return self._hydrate(rows)

    def list_jobs(
        self,
        *,
        states: Iterable[JobState] | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> Page[Job]:
        """List jobs, optionally filtered by state, newest first."""
        params: list[Any] = []
        where = ""
        if states is not None:
            values = [s.value for s in states]
            if not values:
                return Page(items=(), total=0, offset=offset)
            where = f"WHERE state IN ({','.join('?' * len(values))})"
            params.extend(values)

        total = int(self._conn.execute(f"SELECT COUNT(*) FROM jobs {where}", params).fetchone()[0])
        rows = self._conn.execute(
            f"""
            SELECT {_JOB_COLUMNS} FROM jobs {where}
            ORDER BY created_at DESC, seq DESC
            LIMIT ? OFFSET ?
            """,
            [*params, _clamp(limit), max(0, offset)],
        ).fetchall()
        return Page(items=self._hydrate(rows), total=total, offset=max(0, offset))

    # -- sweeps ---------------------------------------------------------------------------

    def create_sweep(self, spec: SweepSpec, jobs: Sequence[JobSpec]) -> tuple[Sweep, list[Job]]:
        """Insert a sweep and its member jobs, in order, as one transaction.

        The members are created inside the same transaction as the sweep row, so a sweep
        can never exist with half its cases -- and, more importantly, the `seq` values they
        receive are consecutive. That is what makes the sweep's queue order identical to
        its on-disk order without storing a second ordering anywhere: the scheduler's
        existing `ORDER BY priority DESC, seq ASC` already produces it.

        Args:
            spec: The sweep's configuration.
            jobs: Member specs, already in the order the cases should run. Each must carry
                the sweep's id and its own position.

        Returns:
            The stored sweep and its member jobs, in order.
        """
        now = self._clock.now()
        with transaction(self._conn):
            self._conn.execute(
                """
                INSERT INTO sweeps (
                    id, name, root, solver, cores_per_job, concurrency, created_at
                ) VALUES (:id, :name, :root, :solver, :cores, :concurrency, :created_at)
                """,
                {
                    "id": spec.sweep_id,
                    "name": spec.name,
                    "root": str(spec.root),
                    "solver": spec.solver,
                    "cores": spec.cores_per_job,
                    "concurrency": spec.concurrency,
                    "created_at": now,
                },
            )
        created = [self.create(job) for job in jobs]
        sweep = self.get_sweep(spec.sweep_id)
        assert sweep is not None
        return sweep, created

    def get_sweep(self, sweep_id: str) -> Sweep | None:
        """One sweep by id, with its live member counts, or ``None``."""
        row = self._conn.execute(
            """
            SELECT id, name, root, solver, cores_per_job, concurrency, created_at
            FROM sweeps WHERE id = ?
            """,
            (sweep_id,),
        ).fetchone()
        if row is None:
            return None
        return self._sweep_from_row(row)

    def sweeps(self) -> Sequence[Sweep]:
        """Every sweep, newest first, with live member counts."""
        rows = self._conn.execute(
            """
            SELECT id, name, root, solver, cores_per_job, concurrency, created_at
            FROM sweeps ORDER BY created_at DESC
            """
        ).fetchall()
        return tuple(self._sweep_from_row(row) for row in rows)

    def sweep_members(self, sweep_id: str) -> Sequence[Job]:
        """A sweep's jobs in sweep order, which is the order they were submitted in."""
        rows = self._conn.execute(
            f"""
            SELECT {_JOB_COLUMNS} FROM jobs
            WHERE sweep_id = ?
            ORDER BY sweep_position ASC, seq ASC
            """,
            (sweep_id,),
        ).fetchall()
        return self._hydrate(rows)

    def active_members_by_sweep(self) -> dict[str, set[str]]:
        """The ids of each sweep's currently active members, keyed by sweep.

        Ids rather than counts, because the scheduler has to union this with the jobs it
        has just admitted -- which are not active in the database yet -- and adding two
        counts would double-count any job that appears in both. Sets make the overlap
        harmless.

        One query for every sweep at once: this runs on every admission pass, and it is
        bounded by the number of running jobs rather than by history.
        """
        rows = self._conn.execute(
            """
            SELECT id, sweep_id FROM jobs
            WHERE sweep_id IS NOT NULL AND state IN ('PREPARING', 'RUNNING')
            """
        ).fetchall()
        active: dict[str, set[str]] = {}
        for row in rows:
            active.setdefault(str(row["sweep_id"]), set()).add(str(row["id"]))
        return active

    def _sweep_from_row(self, row: sqlite3.Row) -> Sweep:
        """Build a sweep, counting its members as they stand right now.

        The counts are derived on read rather than maintained as columns: a stored counter
        has to be updated from every path that changes a job's state, and the first one
        that forgets leaves a sweep permanently believing it has a job running.
        """
        counts = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(state IN ('PREPARING', 'RUNNING')) AS running,
                SUM(state IN ('COMPLETED', 'FAILED', 'CANCELLED', 'REJECTED', 'UNKNOWN'))
                    AS finished
            FROM jobs WHERE sweep_id = ?
            """,
            (row["id"],),
        ).fetchone()
        return Sweep(
            id=str(row["id"]),
            name=str(row["name"]),
            root=Path(str(row["root"])),
            solver=str(row["solver"]),
            cores_per_job=int(row["cores_per_job"]),
            concurrency=int(row["concurrency"]),
            created_at=float(row["created_at"]),
            total=int(counts["total"] or 0),
            running=int(counts["running"] or 0),
            finished=int(counts["finished"] or 0),
        )

    def queue_positions(self) -> dict[str, int]:
        """Map queued job ids to their 1-based position.

        Derived rather than stored (§13.3): a stored column would need rewriting on every
        insert, hold, and priority change, and would drift after a crash.
        """
        rows = self._conn.execute(
            "SELECT id FROM jobs WHERE state = 'QUEUED' ORDER BY priority DESC, seq ASC"
        ).fetchall()
        return {row["id"]: index for index, row in enumerate(rows, start=1)}

    def count_by_state(self) -> dict[JobState, int]:
        """Job counts per state. One grouped query, for the dashboard header."""
        rows = self._conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state").fetchall()
        counts = dict.fromkeys(JobState, 0)
        for row in rows:
            try:
                counts[JobState(row["state"])] = int(row["n"])
            except ValueError:  # pragma: no cover - a state from a future version
                continue
        return counts

    # -- transitions -------------------------------------------------------------------

    def transition(
        self,
        job_id: str,
        target: JobState,
        *,
        detail: str | None = None,
        **updates: Any,
    ) -> Job:
        """Move a job to a new state, atomically and only if the move is legal.

        The ``UPDATE`` names the expected current state in its ``WHERE`` clause, so two
        concurrent attempts cannot both succeed: the loser matches zero rows and raises.
        This is the single point where job state changes, which is what makes the state
        machine in :mod:`dispatch.core.states` an actual guarantee rather than a diagram.

        Args:
            job_id: Job to move.
            target: Destination state.
            detail: Text for the audit trail. Defaults to ``"STATE -> STATE"``.
            **updates: Columns to set in the same statement, restricted to
                :data:`_TRANSITION_FIELDS`.

        Returns:
            The updated job.

        Raises:
            JobNotFound: If the job does not exist.
            IllegalTransition: If the state machine forbids the move, or another writer
                changed the state first.
            ValidationError: On an unknown column in ``updates``.
        """
        unknown = set(updates) - _TRANSITION_FIELDS
        if unknown:
            raise ValidationError(
                f"Cannot set {', '.join(sorted(unknown))} during a transition. "
                f"Settable: {', '.join(sorted(_TRANSITION_FIELDS))}"
            )

        now = self._clock.now()
        with transaction(self._conn):
            row = self._conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            source = JobState(row["state"])
            if not can_transition(source, target):
                raise IllegalTransition(job_id, source, target)

            assignments = ["state = :state"]
            params: dict[str, Any] = {"state": target.value, "id": job_id, "expect": source.value}
            for key, value in updates.items():
                assignments.append(f"{key} = :{key}")
                params[key] = _adapt(value)

            # Timestamps that the state machine owns rather than the caller. Filling them
            # here means every path -- the helpers below, a direct transition, recovery
            # after a restart -- records them identically. COALESCE so an explicit value
            # from the caller always wins and a re-entered timestamp is never overwritten.
            if target is JobState.PREPARING and "started_at" not in updates:
                assignments.append("started_at = COALESCE(started_at, :auto_started)")
                params["auto_started"] = now
            if target in TERMINAL_STATES:
                if "finished_at" not in updates:
                    assignments.append("finished_at = COALESCE(finished_at, :auto_finished)")
                    params["auto_finished"] = now
                if "runtime_s" not in updates:
                    assignments.append(
                        "runtime_s = COALESCE(runtime_s, "
                        "CASE WHEN started_at IS NULL THEN NULL "
                        "ELSE MAX(0, COALESCE(finished_at, :auto_runtime_end) - started_at) END)"
                    )
                    params["auto_runtime_end"] = now

            cursor = self._conn.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = :id AND state = :expect",
                params,
            )
            if cursor.rowcount != 1:
                # Another writer moved the job between the read and the update.
                current = self._conn.execute(
                    "SELECT state FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                raise IllegalTransition(job_id, current["state"] if current else "?", target)

            self._append_event(job_id, "state", detail or f"{source.value} -> {target.value}", now)

        return self.get(job_id)

    def mark_preparing(self, job_id: str) -> Job:
        """Move QUEUED -> PREPARING, stamping ``started_at``.

        A job's clock starts when work on the machine begins, not when the solver does:
        decomposing a large mesh can take twenty minutes, and hiding that from the
        recorded runtime would misreport what the job actually cost.
        """
        return self.transition(job_id, JobState.PREPARING, detail="preparing case")

    def mark_started(
        self, job_id: str, *, pid: int, pid_start_time: float, boot_time: float | None = None
    ) -> Job:
        """Move PREPARING -> RUNNING, recording the supervised process.

        ``pid_start_time`` is stored alongside the pid because a pid alone is not a stable
        identity: after a daemon restart, that number may belong to something else
        entirely. The pair is what makes re-adoption safe (§6.7).

        ``boot_time`` identifies which boot of the machine the job started on, so that a
        later startup can tell "the machine rebooted under this job" from "something killed
        it". ``None`` when the machine cannot report one, which recovery reads as no
        evidence of a reboot rather than as one.
        """
        return self.transition(
            job_id,
            JobState.RUNNING,
            pid=pid,
            pid_start_time=pid_start_time,
            boot_time=boot_time,
            detail=f"solver started, pid {pid}",
        )

    def requeue_for_resume(self, job_id: str, *, detail: str) -> Job:
        """Return an interrupted job to the queue, to be restarted from its own last state.

        Used by startup recovery for a job that was running when the machine rebooted. The
        job keeps its ``seq``, and therefore its exact place in the queue: a reboot must
        not reorder the work that was waiting, and a job that was running before the
        reboot was ahead of everything queued behind it.

        ``started_at``, ``pid`` and ``pid_start_time`` are cleared because they describe a
        process that no longer exists on a machine that no longer exists. Leaving them
        would make the job report a runtime measured across the downtime, and would leave a
        pid that now belongs to something else for the next cancel to signal.
        """
        return self.transition(
            job_id,
            JobState.QUEUED,
            detail=detail,
            resume_requested=1,
            started_at=None,
            pid=None,
            pid_start_time=None,
        )

    def clear_resume_request(self, job_id: str) -> None:
        """Forget that a job asked to resume, once it has been restarted.

        Separate from the transition that starts it so that the flag is cleared exactly
        once the plan has been built with it -- a flag that outlived its restart would ask
        for another one on the job's next pass through the executor.
        """
        with transaction(self._conn):
            self._conn.execute(
                "UPDATE jobs SET resume_requested = 0 WHERE id = ?",
                (job_id,),
            )

    def mark_finished(
        self,
        job_id: str,
        *,
        state: JobState,
        exit_code: int | None = None,
        reason: ExitReason | None = None,
        signal_name: str | None = None,
        finished_at: float | None = None,
        detail_text: str | None = None,
    ) -> Job:
        """Move a job to a terminal state and compute its runtime.

        Runtime is derived here rather than by the caller so that every path -- normal
        exit, cancellation, recovery of a job that finished while the daemon was down --
        records it the same way.

        Args:
            detail_text: Why the job ended, in the solver's words, for a failure. Stored
                as-is; extracting it is the executor's job, because only it knows where
                the log is.
        """
        if state not in TERMINAL_STATES:
            raise ValidationError(f"{state} is not a terminal state")

        when = finished_at if finished_at is not None else self._clock.now()
        row = self._conn.execute("SELECT started_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        started = row["started_at"]
        runtime = max(0.0, when - started) if started is not None else None

        detail = f"{state.value.lower()}"
        if exit_code is not None:
            detail += f", exit code {exit_code}"
        if signal_name:
            detail += f", signal {signal_name}"

        return self.transition(
            job_id,
            state,
            finished_at=when,
            exit_code=exit_code,
            exit_reason=reason,
            exit_signal=signal_name,
            exit_detail=detail_text,
            runtime_s=runtime,
            detail=detail,
        )

    def set_priority(self, job_id: str, priority: int) -> Job:
        """Change a job's priority.

        Not a state transition, so it does not go through :meth:`transition`, but it does
        change scheduling order -- the caller is expected to nudge the scheduler.
        """
        with transaction(self._conn):
            cursor = self._conn.execute(
                "UPDATE jobs SET priority = ? WHERE id = ?", (priority, job_id)
            )
            if cursor.rowcount != 1:
                raise JobNotFound(job_id)
            self._append_event(job_id, "state", f"priority set to {priority}", self._clock.now())
        return self.get(job_id)

    def update_metrics(
        self,
        job_id: str,
        *,
        peak_rss_mb: int | None = None,
        mean_cpu_pct: float | None = None,
    ) -> None:
        """Update measured metrics in place, without a state change.

        ``peak_rss_mb`` only ever increases: ``MAX`` is applied in SQL so that a sample
        arriving out of order cannot lower a peak that really occurred.
        """
        assignments: list[str] = []
        params: dict[str, Any] = {"id": job_id}
        if peak_rss_mb is not None:
            assignments.append("peak_rss_mb = MAX(COALESCE(peak_rss_mb, 0), :rss)")
            params["rss"] = peak_rss_mb
        if mean_cpu_pct is not None:
            assignments.append("mean_cpu_pct = :cpu")
            params["cpu"] = mean_cpu_pct
        if not assignments:
            return
        self._conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = :id", params)

    def set_log_paths(
        self, job_id: str, *, stdout: Path, stderr: Path, log: Path | None = None
    ) -> Job:
        """Record where a job's output will be written.

        Assigned after creation because the paths derive from the job id, which does not
        exist until the row does. The daemon owns the log layout; the repository only
        remembers what it chose.

        Args:
            log: The working-directory log, when there is one. ``None`` leaves the column
                as it was, so a fallback that only moves ``stdout`` cannot silently claim
                a case-directory log that was never written.
        """
        assignments = "stdout_path = ?, stderr_path = ?"
        params: list[Any] = [str(stdout), str(stderr)]
        if log is not None:
            assignments += ", log_path = ?"
            params.append(str(log))
        params.append(job_id)
        cursor = self._conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", params)
        if cursor.rowcount != 1:
            raise JobNotFound(job_id)
        return self.get(job_id)

    def repoint_logs(self, old: Path, new: Path) -> int:
        """Follow a job log that has been rotated aside, for jobs that already finished.

        A second run in the same case directory renames the previous ``log.foam`` out of
        the way (§6.4). Without this, the earlier job's record would point at a path whose
        contents now belong to the newer run, and ``dispatch logs`` on it would show the
        wrong output entirely -- a quiet corruption of the history, which is the one thing
        Dispatch is supposed to be reliable about.

        Only terminal jobs are followed: an active job's log is by definition not the one
        being rotated away, and rewriting a running job's path would point supervision at
        a file nothing is writing to.

        Returns:
            How many job records were updated.
        """
        before, after = str(old), str(new)
        placeholders = ",".join("?" * len(TERMINAL_STATES))
        cursor = self._conn.execute(
            f"""
            UPDATE jobs
               SET log_path    = CASE WHEN log_path    = ? THEN ? ELSE log_path    END,
                   stdout_path = CASE WHEN stdout_path = ? THEN ? ELSE stdout_path END,
                   stderr_path = CASE WHEN stderr_path = ? THEN ? ELSE stderr_path END
             WHERE state IN ({placeholders})
               AND (log_path = ? OR stdout_path = ? OR stderr_path = ?)
            """,
            [
                before,
                after,
                before,
                after,
                before,
                after,
                *sorted(state.value for state in TERMINAL_STATES),
                before,
                before,
                before,
            ],
        )
        return int(cursor.rowcount)

    def update_metadata(self, job_id: str, metadata: CaseMetadata) -> Job:
        """Replace a job's metadata and rebuild its derived indexes."""
        with transaction(self._conn):
            cursor = self._conn.execute(
                "UPDATE jobs SET metadata = ? WHERE id = ?",
                (_dumps(metadata.to_json()), job_id),
            )
            if cursor.rowcount != 1:
                raise JobNotFound(job_id)
            self._write_metadata_index(job_id, metadata)
            self._reindex(job_id)
        return self.get(job_id)

    # -- tags ---------------------------------------------------------------------------

    def set_tags(self, job_id: str, tags: Iterable[str]) -> Job:
        """Replace a job's tags outright."""
        with transaction(self._conn):
            if not self._exists(job_id):
                raise JobNotFound(job_id)
            self._conn.execute("DELETE FROM job_tags WHERE job_id = ?", (job_id,))
            self._write_tags(job_id, normalise_tags(tags))
            self._reindex(job_id)
        return self.get(job_id)

    def edit_tags(self, job_id: str, *, add: Iterable[str] = (), remove: Iterable[str] = ()) -> Job:
        """Add and remove tags in one operation.

        Tags are editable at any point in a job's life, including long after it finished --
        which is usually when you find out a run mattered (§4.5).
        """
        additions = normalise_tags(add)
        removals = normalise_tags(remove)
        with transaction(self._conn):
            if not self._exists(job_id):
                raise JobNotFound(job_id)
            if removals:
                self._conn.execute(
                    f"""
                    DELETE FROM job_tags
                    WHERE job_id = ?
                      AND tag_id IN (SELECT id FROM tags WHERE name IN
                          ({",".join("?" * len(removals))}))
                    """,
                    [job_id, *sorted(removals)],
                )
            self._write_tags(job_id, additions)
            self._reindex(job_id)
        return self.get(job_id)

    def all_tags(self) -> Sequence[tuple[str, int]]:
        """Every tag in use, with its job count, most-used first."""
        rows = self._conn.execute(
            """
            SELECT t.name AS name, COUNT(jt.job_id) AS n
            FROM tags t LEFT JOIN job_tags jt ON jt.tag_id = t.id
            GROUP BY t.id
            HAVING n > 0
            ORDER BY n DESC, t.name ASC
            """
        ).fetchall()
        return tuple((row["name"], int(row["n"])) for row in rows)

    # -- notes, events, samples ----------------------------------------------------------

    def add_note(self, job_id: str, body: str) -> Note:
        """Attach a free-text note to a job."""
        text = body.strip()
        if not text:
            raise ValidationError("A note cannot be empty")
        with transaction(self._conn):
            if not self._exists(job_id):
                raise JobNotFound(job_id)
            note_id = self._insert_note(job_id, text, self._clock.now())
            self._reindex(job_id)
        row = self._conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return note_from_row(row)

    def notes(self, job_id: str) -> Sequence[Note]:
        """A job's notes, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM notes WHERE job_id = ? ORDER BY id ASC", (job_id,)
        ).fetchall()
        return tuple(note_from_row(row) for row in rows)

    def add_event(self, job_id: str, kind: str, detail: str) -> None:
        """Append an audit-trail entry."""
        with transaction(self._conn):
            self._append_event(job_id, kind, detail, self._clock.now())

    def events(self, job_id: str, *, limit: int = 500) -> Sequence[JobEvent]:
        """A job's audit trail, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM job_events WHERE job_id = ? ORDER BY id ASC LIMIT ?",
            (job_id, _clamp(limit)),
        ).fetchall()
        return tuple(event_from_row(row) for row in rows)

    def add_sample(self, job_id: str, sample: Sample) -> None:
        """Record one resource measurement.

        ``INSERT OR REPLACE`` because the primary key is ``(job_id, ts)`` and a duplicate
        timestamp means a redundant sample, not an error worth propagating to the monitor.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO job_samples (job_id, ts, rss_mb, cpu_pct) VALUES (?,?,?,?)",
            (job_id, sample.ts, sample.rss_mb, sample.cpu_pct),
        )

    def samples(self, job_id: str, *, limit: int = 500) -> Sequence[Sample]:
        """A job's most recent samples, oldest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM (
                SELECT ts, rss_mb, cpu_pct FROM job_samples
                WHERE job_id = ? ORDER BY ts DESC LIMIT ?
            ) ORDER BY ts ASC
            """,
            (job_id, _clamp(limit)),
        ).fetchall()
        return tuple(sample_from_row(row) for row in rows)

    # -- provenance -----------------------------------------------------------------------

    def save_provenance(self, job_id: str, prov: Provenance) -> None:
        """Store a job's reproducibility record. Written once, at job start."""
        params = provenance_to_params(job_id, prov)
        columns = ", ".join(params)
        placeholders = ", ".join(f":{key}" for key in params)
        with transaction(self._conn):
            if not self._exists(job_id):
                raise JobNotFound(job_id)
            self._conn.execute(
                f"INSERT OR REPLACE INTO job_provenance ({columns}) VALUES ({placeholders})",
                params,
            )

    def provenance(self, job_id: str) -> Provenance | None:
        """A job's reproducibility record, or ``None`` if it never started."""
        row = self._conn.execute(
            "SELECT * FROM job_provenance WHERE job_id = ?", (job_id,)
        ).fetchone()
        return provenance_from_row(row) if row else None

    # -- deletion --------------------------------------------------------------------------

    def delete(self, job_id: str) -> None:
        """Delete a terminal job and everything attached to it.

        Refuses to delete a job that is queued or running: the row is what the daemon uses
        to find and supervise the process, and removing it would orphan a live simulation.

        Raises:
            ValidationError: If the job has not finished.
        """
        job = self.get(job_id)
        if not job.is_terminal:
            raise ValidationError(
                f"Job {job_id} is {job.state.value} and cannot be deleted. Cancel it first.",
                detail={"state": job.state.value},
            )
        with transaction(self._conn):
            # Attached rows go via ON DELETE CASCADE; the FTS table is not a real child.
            self._conn.execute("DELETE FROM jobs_fts WHERE job_id = ?", (job_id,))
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    # -- search -----------------------------------------------------------------------------

    def search(
        self, query: SearchQuery, *, limit: int = DEFAULT_LIMIT, offset: int = 0
    ) -> Page[Job]:
        """Run a parsed search query.

        Compiles the query into a single parameterised statement. Free text goes to FTS5;
        tags, states, and metadata comparisons become semi-joins against their indexes.
        """
        clauses, params = self._compile(query)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        total = int(
            self._conn.execute(f"SELECT COUNT(*) FROM jobs j {where}", params).fetchone()[0]
        )
        rows = self._conn.execute(
            f"""
            SELECT {_prefixed(_JOB_COLUMNS, "j")} FROM jobs j {where}
            ORDER BY j.created_at DESC, j.seq DESC
            LIMIT ? OFFSET ?
            """,
            [*params, _clamp(limit), max(0, offset)],
        ).fetchall()
        return Page(items=self._hydrate(rows), total=total, offset=max(0, offset))

    def _compile(self, query: SearchQuery) -> tuple[list[str], list[Any]]:
        """Turn a :class:`SearchQuery` into WHERE clauses and bind parameters."""
        clauses: list[str] = []
        params: list[Any] = []

        if query.text:
            clauses.append("j.id IN (SELECT job_id FROM jobs_fts WHERE jobs_fts MATCH ?)")
            params.append(_fts_expression(query.text))

        for tag in sorted(query.tags_include):
            clauses.append(
                "j.id IN (SELECT jt.job_id FROM job_tags jt JOIN tags t ON t.id = jt.tag_id "
                "WHERE t.name = ?)"
            )
            params.append(tag)

        if query.tags_exclude:
            names = sorted(query.tags_exclude)
            clauses.append(
                "j.id NOT IN (SELECT jt.job_id FROM job_tags jt JOIN tags t ON t.id = jt.tag_id "
                f"WHERE t.name IN ({','.join('?' * len(names))}))"
            )
            params.extend(names)

        if query.states:
            values = sorted(s.value for s in query.states)
            clauses.append(f"j.state IN ({','.join('?' * len(values))})")
            params.extend(values)

        if query.states_exclude:
            values = sorted(s.value for s in query.states_exclude)
            clauses.append(f"j.state NOT IN ({','.join('?' * len(values))})")
            params.extend(values)

        if query.solvers:
            values = sorted(query.solvers)
            clauses.append(f"LOWER(j.solver) IN ({','.join('?' * len(values))})")
            params.extend(values)

        if query.resources:
            values = sorted(query.resources)
            clauses.append(f"j.resource_kind IN ({','.join('?' * len(values))})")
            params.extend(values)

        for app in sorted(query.apps):
            clauses.append("j.solver_binary LIKE '%' || ? || '%'")
            params.append(app)

        for name in query.names:
            clauses.append("j.name LIKE '%' || ? || '%'")
            params.append(name)

        for directory in query.dirs:
            clauses.append("j.workdir LIKE '%' || ? || '%'")
            params.append(directory)

        if query.ids:
            ors = " OR ".join("j.id LIKE ? || '%'" for _ in query.ids)
            clauses.append(f"({ors})")
            params.extend(sorted(query.ids))

        if query.created_after is not None:
            clauses.append("j.created_at >= ?")
            params.append(query.created_after)

        if query.created_before is not None:
            clauses.append("j.created_at < ?")
            params.append(query.created_before)

        if query.dirty is not None:
            clauses.append("j.id IN (SELECT job_id FROM job_provenance WHERE git_dirty = ?)")
            params.append(1 if query.dirty else 0)

        for comparison in query.comparisons:
            if comparison.is_metadata:
                clauses.append(
                    "j.id IN (SELECT job_id FROM job_metadata WHERE key = ? "
                    f"AND value_num IS NOT NULL AND value_num {comparison.op.sql} ?)"
                )
                params.extend([comparison.field, comparison.value])
            else:
                # Column names come from a fixed allowlist in the query parser, never
                # from user input, so this interpolation cannot inject.
                clauses.append(f"j.{comparison.field} {comparison.op.sql} ?")
                params.append(comparison.value)

        return clauses, params

    # -- maintenance ---------------------------------------------------------------------------

    def integrity_check(self) -> str:
        """Run SQLite's integrity check. Returns ``"ok"`` on a healthy database."""
        return str(self._conn.execute("PRAGMA integrity_check").fetchone()[0])

    def rebuild_search_index(self) -> int:
        """Rebuild the FTS and metadata indexes for every job.

        Needed only after a manual database edit or a migration that changes how text is
        flattened. Returns the number of jobs reindexed.
        """
        rows = self._conn.execute("SELECT id, metadata FROM jobs").fetchall()
        with transaction(self._conn):
            self._conn.execute("DELETE FROM jobs_fts")
            self._conn.execute("DELETE FROM job_metadata")
            for row in rows:
                metadata = CaseMetadata.from_json(json.loads(row["metadata"] or "{}"))
                self._write_metadata_index(row["id"], metadata)
                self._reindex(row["id"])
        return len(rows)

    def vacuum(self) -> None:
        """Compact the database. Only useful after large deletions."""
        self._conn.execute("VACUUM")

    # -- internals -------------------------------------------------------------------------------

    def _exists(self, job_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row is not None

    def _hydrate(self, rows: Sequence[sqlite3.Row]) -> Sequence[Job]:
        """Attach tags to a batch of job rows with a single query.

        Per-row tag lookups would make a 200-job listing 201 queries; this makes it two.
        """
        if not rows:
            return ()
        ids = [row["id"] for row in rows]
        tag_map: dict[str, set[str]] = {job_id: set() for job_id in ids}
        tag_rows = self._conn.execute(
            f"""
            SELECT jt.job_id AS job_id, t.name AS name
            FROM job_tags jt JOIN tags t ON t.id = jt.tag_id
            WHERE jt.job_id IN ({",".join("?" * len(ids))})
            """,
            ids,
        ).fetchall()
        for row in tag_rows:
            tag_map[row["job_id"]].add(row["name"])
        return tuple(job_from_row(row, tags=frozenset(tag_map[row["id"]])) for row in rows)

    def _tags_for(self, job_id: str) -> frozenset[str]:
        rows = self._conn.execute(
            "SELECT t.name AS name FROM job_tags jt JOIN tags t ON t.id = jt.tag_id "
            "WHERE jt.job_id = ?",
            (job_id,),
        ).fetchall()
        return frozenset(row["name"] for row in rows)

    def _write_tags(self, job_id: str, tags: Iterable[str]) -> None:
        for tag in sorted(set(tags)):
            self._conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            self._conn.execute(
                "INSERT OR IGNORE INTO job_tags (job_id, tag_id) "
                "VALUES (?, (SELECT id FROM tags WHERE name = ?))",
                (job_id, tag),
            )

    def _write_metadata_index(self, job_id: str, metadata: CaseMetadata) -> None:
        """Refresh the derived typed metadata index for one job."""
        self._conn.execute("DELETE FROM job_metadata WHERE job_id = ?", (job_id,))
        spec = self._specs.get(metadata.spec.adapter)
        for key, text, number in metadata.searchable_pairs(spec):
            self._conn.execute(
                "INSERT OR REPLACE INTO job_metadata (job_id, key, value_text, value_num) "
                "VALUES (?,?,?,?)",
                (job_id, key, text, number),
            )

    def _reindex(self, job_id: str) -> None:
        """Rebuild one job's FTS row from all four contributing sources.

        Called explicitly rather than by triggers because the row draws on ``jobs``,
        ``notes``, and ``job_tags``: a trigger web spanning three tables would be harder
        to follow than one call at the single place writes happen.
        """
        row = self._conn.execute(
            "SELECT name, workdir, solver, solver_binary, metadata FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return
        metadata = CaseMetadata.from_json(json.loads(row["metadata"] or "{}"))
        notes = " ".join(
            r["body"]
            for r in self._conn.execute("SELECT body FROM notes WHERE job_id = ?", (job_id,))
        )
        tags = " ".join(sorted(self._tags_for(job_id)))

        self._conn.execute("DELETE FROM jobs_fts WHERE job_id = ?", (job_id,))
        self._conn.execute(
            """
            INSERT INTO jobs_fts (job_id, name, workdir, solver, solver_binary, metadata,
                                  notes, tags)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                job_id,
                row["name"],
                row["workdir"],
                row["solver"],
                row["solver_binary"] or "",
                metadata.flatten_for_fts(),
                notes,
                tags,
            ),
        )

    def _insert_note(self, job_id: str, body: str, ts: float) -> int:
        cursor = self._conn.execute(
            "INSERT INTO notes (job_id, ts, body) VALUES (?,?,?)", (job_id, ts, body)
        )
        return int(cursor.lastrowid or 0)

    def _append_event(self, job_id: str, kind: str, detail: str, ts: float) -> None:
        self._conn.execute(
            "INSERT INTO job_events (job_id, ts, kind, detail) VALUES (?,?,?,?)",
            (job_id, ts, kind, detail),
        )


# -- helpers ------------------------------------------------------------------------------------


def _clamp(limit: int) -> int:
    """Bound a page size. Protects both daemon memory and IPC message size."""
    return max(1, min(int(limit), MAX_LIMIT))


def _dumps(payload: Any) -> str:
    """Compact JSON. Separators matter: at thousands of rows the spaces add up."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _adapt(value: Any) -> Any:
    """Convert a domain value to something sqlite3 can bind."""
    if isinstance(value, ExitReason):
        return value.value
    if isinstance(value, JobState):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _prefixed(columns: str, alias: str) -> str:
    """Qualify a column list with a table alias."""
    return ", ".join(f"{alias}.{name.strip()}" for name in columns.split(",") if name.strip())


def _fts_expression(text: str) -> str:
    """Build a safe FTS5 MATCH expression from free-text terms.

    Each term is quoted and given a prefix wildcard, so ``naca`` finds ``naca0018`` and a
    term containing FTS5 operator characters (``-``, ``*``, ``:``, ``(``) is treated as
    text rather than as syntax. Without this, a user searching for ``re-100`` would get a
    syntax error instead of results.
    """
    terms = []
    for term in text.split():
        escaped = term.replace('"', '""')
        terms.append(f'"{escaped}"*')
    return " ".join(terms)


TRANSITION_FIELDS: Final = _TRANSITION_FIELDS
"""Public alias: columns a state transition may set."""
