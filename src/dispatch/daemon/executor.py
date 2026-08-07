"""Running the plans that adapters describe.

The executor is the only component that starts processes, and it does so without knowing
what any of them are. It asks the adapter for an :class:`~dispatch.core.plan.ExecutionPlan`,
runs the preparation steps, spawns the solve step, waits, and records the outcome.

Mesh decomposition, source compilation, and mesh reconstruction are all just preparation
steps; the executor cannot tell them apart. That is the entire payoff of the adapter
design, and it is why adding a solver that must be *compiled* before it runs required no
change here at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from dispatch.adapters.base import CaseContext, SolverAdapter, generic_failure_summary
from dispatch.adapters.registry import AdapterRegistry
from dispatch.core.clock import Clock, SystemClock
from dispatch.core.config import Config
from dispatch.core.errors import AdapterError, DispatchError
from dispatch.core.models import Job
from dispatch.core.plan import CommandStep, ExecutionPlan, StepOutcome
from dispatch.core.states import ExitReason, JobState
from dispatch.daemon.events import EventBus
from dispatch.daemon.monitor import read_tail
from dispatch.daemon.process import (
    ProcessHandle,
    ProcessManager,
    SpawnError,
    describe_exit,
    is_alive,
    matches_start_time,
)
from dispatch.daemon.provenance import ProvenanceCollector
from dispatch.daemon.resources import ResourceModel
from dispatch.db.repository import JobRepository
from dispatch.ipc.protocol import Event, encode_job

__all__ = ["JobExecutor", "RunningJob"]

log = logging.getLogger(__name__)

FAILURE_TAIL_BYTES = 16384
"""How much of each log to read when explaining a failure.

Larger than the progress tail: a solver's fatal error block and stack trace, or a parallel
launcher's boxed complaint, run to well over a hundred lines.
"""


@dataclass
class RunningJob:
    """Bookkeeping for one in-flight job."""

    job_id: str
    task: asyncio.Task[None]
    handle: ProcessHandle | None = None
    adapter: SolverAdapter | None = None
    context: CaseContext | None = None
    cancelling: bool = False
    force_kill: bool = False
    log_files: list[IO[bytes]] = field(default_factory=list)

    def close_logs(self) -> None:
        for handle in self.log_files:
            with contextlib.suppress(Exception):
                handle.close()
        self.log_files.clear()


class JobExecutor:
    """Turns admitted jobs into supervised processes.

    Args:
        repo: Persistence.
        registry: Solver adapters.
        resources: The ledger, released when a job ends.
        config: Daemon configuration.
        bus: Event bus, for pushing state changes to clients.
        clock: Time source.
        provenance: Reproducibility collector.
        processes: Process primitives, injectable for tests.
        on_finished: Called after every terminal transition, so the scheduler can
            reconsider the queue. Injected rather than imported to keep the dependency
            pointing one way.
    """

    def __init__(
        self,
        *,
        repo: JobRepository,
        registry: AdapterRegistry,
        resources: ResourceModel,
        config: Config,
        bus: EventBus,
        clock: Clock | None = None,
        provenance: ProvenanceCollector | None = None,
        processes: ProcessManager | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        self._repo = repo
        self._registry = registry
        self._resources = resources
        self._config = config
        self._bus = bus
        self._clock = clock or SystemClock()
        self._provenance = provenance or ProvenanceCollector(clock=self._clock)
        self._processes = processes or ProcessManager(clock=self._clock)
        self._on_finished = on_finished
        self._running: dict[str, RunningJob] = {}

    @property
    def running_ids(self) -> list[str]:
        """Ids of jobs this executor is currently supervising."""
        return list(self._running)

    def is_running(self, job_id: str) -> bool:
        """Whether this executor is supervising the given job."""
        return job_id in self._running

    # -- launching ---------------------------------------------------------------------------

    def launch(self, job: Job) -> None:
        """Begin running a job.

        Returns immediately; the work proceeds in a task. The resource allocation is taken
        by the *scheduler* before this is called, so that admitting and allocating cannot
        interleave with another pass.
        """
        if job.id in self._running:
            log.warning("Job %s is already running; ignoring duplicate launch", job.id)
            return
        task = asyncio.create_task(self._run(job), name=f"dispatch-job-{job.id[:8]}")
        self._running[job.id] = RunningJob(job_id=job.id, task=task)

    async def _run(self, job: Job) -> None:
        """Drive one job from admitted to terminal. Never raises."""
        entry = self._running[job.id]
        try:
            await self._execute(job, entry)
        except asyncio.CancelledError:
            await self._finish(job.id, JobState.CANCELLED, reason=ExitReason.CANCELLED)
            raise
        except DispatchError as exc:
            log.error("Job %s failed: %s", job.id, exc)
            self._event(job.id, "warn", str(exc))
            await self._finish(job.id, JobState.FAILED, reason=ExitReason.PREPARE_FAILED)
        except Exception as exc:
            log.exception("Unexpected failure running job %s", job.id)
            self._event(job.id, "warn", f"internal error: {exc}")
            await self._finish(job.id, JobState.FAILED, reason=ExitReason.PREPARE_FAILED)
        finally:
            # In the `finally` so that every exit -- solved, cancelled mid-prepare, or
            # crashed -- leaves the case as its owner wrote it. Adapters may edit a case to
            # steer the run they are supervising, and an edit that outlives its run silently
            # changes what every later run of that case does.
            self._finalize(entry)
            entry.close_logs()
            self._running.pop(job.id, None)
            self._resources.release(job.id)
            if self._on_finished:
                self._on_finished()

    async def _execute(self, job: Job, entry: RunningJob) -> None:
        adapter = self._registry.get(job.solver)
        entry.adapter = adapter

        ctx = self._build_context(job, adapter)
        entry.context = ctx

        plan = await asyncio.to_thread(adapter.plan, ctx)
        if not isinstance(plan, ExecutionPlan):  # pragma: no cover - adapter contract
            raise AdapterError(f"Adapter {job.solver!r} did not return an execution plan")

        self._transition(job.id, JobState.PREPARING, detail="preparing case")

        log_dir = self._log_dir(job)
        steps_log = log_dir / "steps.log"

        for step in plan.prepare:
            if entry.cancelling:
                await self._finish(job.id, JobState.CANCELLED, reason=ExitReason.CANCELLED)
                return
            outcome = await self._run_step(step, steps_log, entry)
            if outcome.fatal:
                detail = (
                    f"{step.description} timed out"
                    if outcome.timed_out
                    else f"{step.description} failed with exit code {outcome.exit_code}"
                )
                self._event(job.id, "step", detail)
                # A preparation step is where the most explicable failures happen -- a
                # decomposition that will not divide, a source file that will not compile.
                # Its output went to the step transcript, not the solver's log.
                reason_text = await asyncio.to_thread(
                    self._explain_failure, job.id, entry, steps_log
                )
                await self._finish(
                    job.id,
                    JobState.FAILED,
                    exit_code=outcome.exit_code,
                    reason=ExitReason.TIMEOUT if outcome.timed_out else ExitReason.PREPARE_FAILED,
                    detail_text=reason_text,
                )
                return

        if entry.cancelling:
            await self._finish(job.id, JobState.CANCELLED, reason=ExitReason.CANCELLED)
            return

        await self._run_solve(job, entry, plan, ctx, adapter, log_dir)

    async def _run_solve(
        self,
        job: Job,
        entry: RunningJob,
        plan: ExecutionPlan,
        ctx: CaseContext,
        adapter: SolverAdapter,
        log_dir: Path,
    ) -> None:
        """Spawn the solve step, record provenance, and wait for it to finish."""
        step = plan.solve
        exit_file = log_dir / "exit_code"
        with contextlib.suppress(OSError):
            exit_file.unlink()

        # Provenance is captured here -- after preparation, immediately before the spawn --
        # because this is the moment its answers are true.
        record = await self._provenance.capture(
            ctx=ctx,
            adapter=adapter,
            argv=step.argv,
            env=step.env or os.environ,
            warn=lambda message: self._event(job.id, "warn", message),
        )

        stdout = open(job.stdout_path or (log_dir / "stdout.log"), "ab", buffering=0)  # noqa: SIM115
        stderr = open(job.stderr_path or (log_dir / "stderr.log"), "ab", buffering=0)  # noqa: SIM115
        entry.log_files.extend([stdout, stderr])

        try:
            handle = await self._processes.spawn(
                step, stdout=stdout, stderr=stderr, exit_file=exit_file
            )
        except SpawnError as exc:
            self._event(job.id, "warn", str(exc))
            await self._finish(job.id, JobState.FAILED, reason=ExitReason.PREPARE_FAILED)
            return

        entry.handle = handle
        self._repo.save_provenance(job.id, record)
        self._repo.mark_started(job.id, pid=handle.pid, pid_start_time=handle.start_time)
        self._publish(job.id)
        log.info("Job %s started: %s (pid %d)", job.id[:8], step.render(), handle.pid)

        code = await self._processes.wait(handle, timeout=step.timeout_s)
        timed_out = code is None and step.timeout_s is not None
        if timed_out:
            self._event(job.id, "warn", f"solver exceeded its {step.timeout_s:.0f}s timeout")
            await self._processes.terminate(handle, grace_s=self._config.daemon.cancel_grace_s)

        if code is None:
            code = ProcessManager.read_exit_file(exit_file)

        for step in plan.cleanup:
            await self._run_step(step, log_dir / "steps.log", entry)

        await self._conclude(job.id, entry, code, timed_out=timed_out)

    def _finalize(self, entry: RunningJob) -> None:
        """Let the adapter undo whatever it did to steer this run.

        Synchronous on purpose. This is called from a ``finally`` that runs while the
        supervising task is being cancelled, and an ``await`` there would raise
        ``CancelledError`` again before the work happened -- so the one path that most
        needs the case restored is exactly the path that would skip it. The contract is
        that :meth:`~dispatch.adapters.base.BaseAdapter.finalize` is a couple of file
        operations; the default is a no-op.

        Never allowed to affect the outcome: a job that ran is a job that ran, whatever
        happens while tidying up after it.
        """
        if entry.adapter is None or entry.context is None:
            return
        try:
            entry.adapter.finalize(entry.context)
        except Exception as exc:
            log.warning("Adapter finalize failed for job %s: %s", entry.job_id, exc)

    async def _conclude(
        self, job_id: str, entry: RunningJob, code: int | None, *, timed_out: bool
    ) -> None:
        """Classify how a solver ended and record it."""
        exit_code, signal_name = describe_exit(code)

        if entry.cancelling:
            state, reason = JobState.CANCELLED, ExitReason.CANCELLED
        elif timed_out:
            state, reason = JobState.FAILED, ExitReason.TIMEOUT
        elif code is None:
            state, reason = JobState.UNKNOWN, ExitReason.LOST
        elif code == 0:
            state, reason = JobState.COMPLETED, ExitReason.OK
        elif signal_name:
            state = JobState.FAILED
            reason = ExitReason.OOM if _was_oom_killed(signal_name, entry) else ExitReason.SIGNAL
        else:
            state, reason = JobState.FAILED, ExitReason.NONZERO

        detail_text = None
        if state is JobState.FAILED:
            detail_text = await asyncio.to_thread(self._explain_failure, job_id, entry)

        await self._finish(
            job_id,
            state,
            exit_code=exit_code,
            reason=reason,
            signal_name=signal_name,
            detail_text=detail_text,
        )

    def _explain_failure(
        self, job_id: str, entry: RunningJob, only: Path | None = None
    ) -> str | None:
        """Read the end of a failed job's logs and ask its adapter what went wrong.

        stderr first: when a run dies before the solver starts -- a missing library, an MPI
        launcher refusing the rank count -- stdout is empty and the whole explanation is on
        stderr. Both are read because the reverse is also true, and solvers that write
        their fatal errors to stdout are common.

        Args:
            only: Read just this file. Used for a preparation failure, whose output is in
                the step transcript and whose solver logs are empty and misleading.
        """
        job = self._repo.get_optional(job_id)
        if job is None:
            return None

        log_dir = self._config.paths.job_dir(job.id)
        sources = (
            [only]
            if only is not None
            else [
                job.stderr_path or (log_dir / "stderr.log"),
                job.stdout_path or (log_dir / "stdout.log"),
                log_dir / "steps.log",
            ]
        )
        tail = "\n".join(
            part for part in (read_tail(path, FAILURE_TAIL_BYTES) for path in sources) if part
        )
        if not tail.strip():
            return None

        adapter, ctx = entry.adapter, entry.context
        if adapter is None or ctx is None:
            return generic_failure_summary(tail)
        try:
            return adapter.explain_failure(tail, ctx)
        except Exception as exc:
            log.debug("Adapter %s could not explain the failure: %s", job.solver, exc)
            return generic_failure_summary(tail)

    # -- steps ------------------------------------------------------------------------------------

    async def _run_step(
        self, step: CommandStep, transcript: Path, entry: RunningJob
    ) -> StepOutcome:
        """Run one preparation or cleanup step, appending to the step transcript."""
        started = self._clock.monotonic()
        self._event(entry.job_id, "step", step.description)
        log.info("Job %s: %s", entry.job_id[:8], step.render())

        transcript.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript, "ab", buffering=0) as sink:
            sink.write(f"\n$ {step.render()}\n".encode())
            try:
                handle = await self._processes.spawn(step, stdout=sink, stderr=sink)
            except SpawnError as exc:
                sink.write(f"dispatch: {exc}\n".encode())
                return StepOutcome(
                    step=step, exit_code=127, duration_s=self._clock.monotonic() - started
                )

            entry.handle = handle
            code = await self._processes.wait(handle, timeout=step.timeout_s)
            timed_out = code is None and step.timeout_s is not None
            if timed_out:
                await self._processes.terminate(handle, grace_s=self._config.daemon.cancel_grace_s)
                sink.write(b"dispatch: step timed out\n")

        entry.handle = None
        return StepOutcome(
            step=step,
            exit_code=code,
            duration_s=self._clock.monotonic() - started,
            timed_out=timed_out,
        )

    # -- cancellation ------------------------------------------------------------------------------

    async def cancel(self, job_id: str, *, force: bool = False) -> bool:
        """Stop a running job.

        Tries the adapter's native clean stop first when one exists -- for a CFD solver
        that usually means "write the current state and exit", which leaves a usable
        result instead of a truncated one. Falls back to the signal ladder.

        Returns:
            ``True`` if the job was running and has been asked to stop.
        """
        entry = self._running.get(job_id)
        if entry is None:
            return False

        entry.cancelling = True
        entry.force_kill = force
        self._event(job_id, "signal", "cancellation requested" + (" (forced)" if force else ""))

        if not force and entry.adapter is not None and entry.context is not None:
            try:
                if await asyncio.to_thread(entry.adapter.stop_gracefully, entry.context):
                    self._event(job_id, "signal", "asked the solver to stop cleanly")
                    return True
            except Exception as exc:
                log.warning("Graceful stop failed for job %s: %s", job_id, exc)

        if entry.handle is not None:
            grace = 0.0 if force else self._config.daemon.cancel_grace_s
            await self._processes.terminate(entry.handle, grace_s=grace)
        else:
            entry.task.cancel()
        return True

    async def shutdown(self, *, kill_jobs: bool = False) -> None:
        """Stop supervising.

        By default the running simulations are **left alone**: stopping Dispatch and
        stopping a week-long run are different actions, and conflating them would be
        unforgivable. The processes are in their own sessions, so they survive, and
        recovery re-adopts them at the next start.
        """
        tasks = [entry.task for entry in self._running.values()]

        if kill_jobs:
            await asyncio.gather(
                *(self.cancel(job_id, force=True) for job_id in list(self._running)),
                return_exceptions=True,
            )
        else:
            for entry in self._running.values():
                entry.task.cancel()
            for entry in self._running.values():
                entry.close_logs()
            self._running.clear()

        # Wait either way. A supervising task still running after this returns would try to
        # record its job's outcome against a database the caller is about to close -- so
        # the kill would be real but the record of it would not.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- re-adoption -------------------------------------------------------------------------------

    def adopt(self, job: Job) -> bool:
        """Resume supervising a job that outlived a previous daemon.

        Returns ``False`` if the process is gone or the pid now belongs to something else.
        """
        if job.pid is None or not is_alive(job.pid):
            return False
        if not matches_start_time(job.pid, job.pid_start_time):
            return False

        handle = ProcessHandle(
            pid=job.pid,
            start_time=job.pid_start_time or 0.0,
            process=None,
            exit_file=self._log_dir(job) / "exit_code",
        )
        task = asyncio.create_task(
            self._supervise_adopted(job, handle), name=f"dispatch-adopt-{job.id[:8]}"
        )
        entry = RunningJob(job_id=job.id, task=task, handle=handle)
        with contextlib.suppress(AdapterError):
            entry.adapter = self._registry.get(job.solver)
            entry.context = self._build_context(job, entry.adapter)
        self._running[job.id] = entry
        self._resources.acquire(job.id, job.resources)
        log.info("Re-adopted job %s (pid %d)", job.id[:8], job.pid)
        return True

    async def _supervise_adopted(self, job: Job, handle: ProcessHandle) -> None:
        """Wait for a re-adopted process and record its outcome from the sentinel."""
        entry = self._running[job.id]
        try:
            await self._processes.wait_adopted(handle)
            code = ProcessManager.read_exit_file(handle.exit_file) if handle.exit_file else None
            await self._conclude(job.id, entry, code, timed_out=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Failed while supervising re-adopted job %s", job.id)
            await self._finish(job.id, JobState.UNKNOWN, reason=ExitReason.LOST)
        finally:
            self._finalize(entry)
            entry.close_logs()
            self._running.pop(job.id, None)
            self._resources.release(job.id)
            if self._on_finished:
                self._on_finished()

    # -- helpers -----------------------------------------------------------------------------------

    def _build_context(self, job: Job, adapter: SolverAdapter) -> CaseContext:
        """Assemble the case context, including the adapter's own environment."""
        entry_path = job.metadata.extra.get("entry") if job.metadata else None
        base = self._registry.context(
            job.workdir,
            cores=job.cores,
            ram_mb=job.ram_estimate_mb,
            entry=Path(str(entry_path)) if entry_path else None,
            env=dict(os.environ),
            adapter=job.solver,
            job_name=job.name,
        )
        try:
            env = adapter.prepare_environment(base)
        except Exception as exc:
            self._event(job.id, "warn", f"could not prepare the solver environment: {exc}")
            env = base.env
        return CaseContext(
            workdir=base.workdir,
            cores=base.cores,
            ram_mb=base.ram_mb,
            entry=base.entry,
            env=dict(env),
            settings=base.settings,
            job_name=base.job_name,
            metadata=job.metadata,
        )

    def _log_dir(self, job: Job) -> Path:
        directory = self._config.paths.job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _transition(self, job_id: str, state: JobState, *, detail: str | None = None) -> None:
        self._repo.transition(job_id, state, detail=detail)
        self._publish(job_id)

    async def _finish(
        self,
        job_id: str,
        state: JobState,
        *,
        exit_code: int | None = None,
        reason: ExitReason | None = None,
        signal_name: str | None = None,
        detail_text: str | None = None,
    ) -> None:
        """Record a terminal outcome, tolerating a job that already reached one."""
        try:
            self._repo.mark_finished(
                job_id,
                state=state,
                exit_code=exit_code,
                reason=reason,
                signal_name=signal_name,
                detail_text=detail_text,
            )
        except DispatchError as exc:
            log.debug("Job %s was already terminal: %s", job_id, exc)
            return
        self._publish(job_id)
        log.info("Job %s finished: %s (exit %s)", job_id[:8], state.value, exit_code)

    def _publish(self, job_id: str) -> None:
        job = self._repo.get_optional(job_id)
        if job is not None:
            self._bus.publish(Event.JOB_STATE, encode_job(job))

    def _event(self, job_id: str, kind: str, detail: str) -> None:
        with contextlib.suppress(DispatchError):
            self._repo.add_event(job_id, kind, detail)


def _was_oom_killed(signal_name: str, entry: RunningJob) -> bool:
    """Distinguish an out-of-memory kill from a deliberate SIGKILL.

    Worth separating: OOM means the RAM estimate was wrong, which is actionable next time,
    whereas a plain SIGKILL usually means somebody ran ``kill -9``.
    """
    if signal_name != "KILL":
        return False
    if entry.cancelling:
        return False
    return _oom_recently(entry.handle.pid if entry.handle else None)


def _oom_recently(pid: int | None) -> bool:
    """Check the cgroup OOM counter for this process's group.

    Reads ``memory.events`` rather than parsing ``dmesg``, which needs privileges Dispatch
    does not have and should not want.
    """
    if pid is None:
        return False
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    relative = cgroup.rpartition(":")[2].lstrip("/")
    events = Path("/sys/fs/cgroup") / relative / "memory.events"
    try:
        for line in events.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            if key in ("oom_kill", "oom_group_kill") and value.strip() not in ("", "0"):
                return True
    except OSError:
        return False
    return False


def summarise_environment(env: Mapping[str, str], keys: Any) -> dict[str, str]:
    """Filter an environment down to declared keys, for display."""
    return {key: env[key] for key in keys if key in env}
