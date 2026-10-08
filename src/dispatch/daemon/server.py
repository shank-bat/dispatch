"""The Unix socket server and request dispatch.

One handler per method, a table mapping names to handlers, and a session object per
connection. Handlers are synchronous where the work is synchronous -- most of them are a
repository call and an encode -- because wrapping fast local work in coroutines buys
nothing and costs readability.

The socket is created with mode ``0600``. On a single-user machine that *is* the security
model: there are no users to authenticate and no network to protect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from dispatch.adapters.base import DEFAULT_LOG_NAME
from dispatch.adapters.registry import AdapterRegistry
from dispatch.core.clock import Clock, SystemClock
from dispatch.core.config import Config, CpuMode
from dispatch.core.errors import DispatchError, ValidationError
from dispatch.core.metadata import CaseMetadata
from dispatch.core.models import Job, JobSpec, ResourceRequest, SweepSpec
from dispatch.core.query import parse_query
from dispatch.core.series import PlotData
from dispatch.core.states import JobState
from dispatch.core.visual import VisualKind, VisualRequest
from dispatch.daemon.dryrun import CaseInspector
from dispatch.daemon.events import EventBus, Subscription
from dispatch.daemon.executor import JobExecutor
from dispatch.daemon.joblog import assign_log_paths
from dispatch.daemon.monitor import SystemMonitor
from dispatch.daemon.plotdata import extract_datasets, latest_case_values
from dispatch.daemon.projects import search_projects
from dispatch.daemon.renders import RenderManager
from dispatch.daemon.resources import CPU_MODE_SETTING, ResourceModel
from dispatch.daemon.scheduler import Scheduler
from dispatch.db.repository import JobRepository
from dispatch.ipc.codec import ProtocolError, read_message, write_message
from dispatch.ipc.protocol import (
    PROTOCOL_VERSION,
    Event,
    Method,
    Response,
    encode_case_report,
    encode_dataset,
    encode_detection,
    encode_dry_run,
    encode_event,
    encode_job,
    encode_metadata_spec,
    encode_note,
    encode_page,
    encode_plot_data,
    encode_project_hit,
    encode_provenance,
    encode_report,
    encode_sample,
    encode_snapshot,
    encode_sweep,
)
from dispatch.ipc.socketpath import start_unix_server
from dispatch.version import __version__

__all__ = ["IpcServer"]

log = logging.getLogger(__name__)

Handler = Callable[["ClientSession", dict[str, Any]], Any]


class ClientSession:
    """One connected client."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        subscription: Subscription,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.subscription = subscription
        self.greeted = False
        self.client = "unknown"


class IpcServer:
    """Serves the daemon protocol on a Unix socket.

    Args:
        config: Daemon configuration, for the socket path.
        repo: Persistence.
        scheduler: Queue operations.
        executor: Cancellation and running state.
        resources: The ledger.
        registry: Solver adapters.
        inspector: Detection, validation, and dry run.
        monitor: System sampling, for on-demand snapshots.
        bus: Event bus.
        clock: Time source.
        on_shutdown: Called when a client requests daemon shutdown.
    """

    def __init__(
        self,
        *,
        config: Config,
        repo: JobRepository,
        scheduler: Scheduler,
        executor: JobExecutor,
        resources: ResourceModel,
        registry: AdapterRegistry,
        inspector: CaseInspector,
        monitor: SystemMonitor,
        bus: EventBus,
        clock: Clock | None = None,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._repo = repo
        self._scheduler = scheduler
        self._executor = executor
        self._resources = resources
        self._registry = registry
        self._inspector = inspector
        self._monitor = monitor
        self._bus = bus
        self._clock = clock or SystemClock()
        self._on_shutdown = on_shutdown

        self._server: asyncio.Server | None = None
        self._sessions: set[ClientSession] = set()
        self._started_at = self._clock.now()
        self._renders = RenderManager(bus)
        self._handlers: dict[str, Handler] = self._build_handlers()

    # -- lifecycle ------------------------------------------------------------------------

    async def start(self) -> None:
        """Bind the socket and begin accepting connections."""
        path = self._config.paths.socket
        path.parent.mkdir(parents=True, exist_ok=True)
        # A socket left behind by a crashed daemon would make bind() fail. The pidfile
        # lock has already established that no other daemon is running, so removing it is
        # safe here and nowhere else.
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

        self._server = await start_unix_server(self._handle_client, path)
        path.chmod(0o600)
        log.info("Listening on %s", path)

    async def stop(self) -> None:
        """Tell clients the daemon is going away, stop renders, then close the socket.

        Renders are stopped rather than left running: unlike a simulation, a renderer is not
        re-adopted on restart, so one left behind would be an orphan using the machine with
        nothing left to report its result to.
        """
        await self._renders.shutdown()
        self._bus.publish(Event.DAEMON_SHUTDOWN, {"reason": "the daemon is shutting down"})
        await asyncio.sleep(0)  # let the notification reach session outboxes

        sessions = list(self._sessions)
        self._sessions.clear()
        for session in sessions:
            session.writer.close()
        # Await each close rather than dropping the writers: a StreamWriter garbage
        # collected with a live transport raises from __del__, which on a daemon that
        # restarts under load means noise in the log at exactly the wrong moment.
        for session in sessions:
            with contextlib.suppress(Exception):
                await session.writer.wait_closed()

        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        with contextlib.suppress(FileNotFoundError):
            self._config.paths.socket.unlink()

    # -- connections -----------------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        session = ClientSession(reader, writer, self._bus.subscribe())
        self._sessions.add(session)
        pump = asyncio.create_task(self._pump_events(session), name="dispatch-session-pump")
        try:
            await self._serve(session)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            log.exception("Client session failed")
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
            self._bus.unsubscribe(session.subscription)
            self._sessions.discard(session)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _serve(self, session: ClientSession) -> None:
        """Read and answer requests until the client goes away."""
        while True:
            try:
                message = await read_message(session.reader)
            except ProtocolError as exc:
                await self._send(session, Response.failure(0, "PROTOCOL_ERROR", str(exc)))
                return
            if message is None:
                return

            request_id = _as_int(message.get("id"))
            method = str(message.get("method", ""))
            params = message.get("params") or {}
            if not isinstance(params, dict):
                await self._send(
                    session,
                    Response.failure(request_id, "PROTOCOL_ERROR", "params must be an object"),
                )
                continue

            await self._send(session, await self._dispatch(session, request_id, method, params))

    async def _dispatch(
        self, session: ClientSession, request_id: int, method: str, params: dict[str, Any]
    ) -> Response:
        """Route one request to its handler, converting errors into responses."""
        handler = self._handlers.get(method)
        if handler is None:
            return Response.failure(
                request_id,
                "UNKNOWN_METHOD",
                f"No such method {method!r}. Known methods: {', '.join(sorted(self._handlers))}",
            )
        if not session.greeted and method != Method.HELLO:
            return Response.failure(request_id, "NOT_GREETED", "Send `hello` before anything else")

        try:
            result = handler(session, params)
            if isinstance(result, Awaitable):
                result = await result
        except DispatchError as exc:
            return Response.failure(request_id, exc.code, exc.message, **exc.detail)
        except Exception as exc:
            log.exception("Handler for %s failed", method)
            return Response.failure(request_id, "INTERNAL_ERROR", str(exc))
        return Response(id=request_id, ok=True, result=result)

    async def _send(self, session: ClientSession, response: Response) -> None:
        with contextlib.suppress(ConnectionResetError, BrokenPipeError, ProtocolError):
            await write_message(session.writer, response.to_json())

    async def _pump_events(self, session: ClientSession) -> None:
        """Forward this session's queued events to its socket."""
        try:
            while True:
                notification = await session.subscription.queue.get()
                await write_message(session.writer, notification.to_json())
        except asyncio.CancelledError:
            raise
        except (ConnectionResetError, BrokenPipeError, ProtocolError):
            pass

    # -- handlers -------------------------------------------------------------------------------

    def _build_handlers(self) -> dict[str, Handler]:
        return {
            Method.HELLO: self._hello,
            Method.DAEMON_INFO: self._daemon_info,
            Method.DAEMON_SHUTDOWN: self._daemon_shutdown,
            Method.SYSTEM_SNAPSHOT: self._system_snapshot,
            Method.SCHEDULER_CPU_MODE: self._scheduler_cpu_mode,
            Method.JOB_SUBMIT: self._job_submit,
            Method.SWEEP_SUBMIT: self._sweep_submit,
            Method.SWEEP_LIST: self._sweep_list,
            Method.JOB_LIST: self._job_list,
            Method.JOB_GET: self._job_get,
            Method.JOB_CANCEL: self._job_cancel,
            Method.JOB_HOLD: self._job_hold,
            Method.JOB_RELEASE: self._job_release,
            Method.JOB_PRIORITY: self._job_priority,
            Method.JOB_DELETE: self._job_delete,
            Method.JOB_NOTE: self._job_note,
            Method.JOB_TAG: self._job_tag,
            Method.JOB_PROVENANCE: self._job_provenance,
            Method.JOB_REPARTITION: self._job_repartition,
            Method.JOB_METRICS: self._job_metrics,
            Method.JOB_SERIES: self._job_series,
            Method.TAGS_LIST: self._tags_list,
            Method.HISTORY_SEARCH: self._history_search,
            Method.CASE_DETECT: self._case_detect,
            Method.CASE_VALIDATE: self._case_validate,
            Method.CASE_INFO: self._case_info,
            Method.CASE_RENDER: self._case_render,
            Method.CASE_FIELDS: self._case_fields,
            Method.RENDER_LIST: self._render_list,
            Method.RENDER_CANCEL: self._render_cancel,
            Method.CASE_DRYRUN: self._case_dryrun,
            Method.FS_LIST: self._fs_list,
            Method.PROJECTS_SEARCH: self._projects_search,
            Method.SUBSCRIBE: self._subscribe,
            Method.UNSUBSCRIBE: self._unsubscribe,
        }

    def _hello(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        """Handshake. A version mismatch must say "restart the daemon", not raise."""
        client_protocol = _as_int(params.get("protocol"))
        if client_protocol != PROTOCOL_VERSION:
            raise ValidationError(
                f"This client speaks protocol version {client_protocol} but the running "
                f"daemon speaks version {PROTOCOL_VERSION}. Restart the daemon "
                "(`systemctl --user restart dispatchd`, or kill it and rerun) so both "
                "sides come from the same install.",
                detail={"client": client_protocol, "daemon": PROTOCOL_VERSION},
            )
        session.greeted = True
        session.client = str(params.get("client", "unknown"))
        return {
            "protocol": PROTOCOL_VERSION,
            "version": __version__,
            "hostname": os.uname().nodename,
            "started_at": self._started_at,
            "adapters": list(self._registry.names),
        }

    def _daemon_info(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": __version__,
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "uptime_s": self._clock.now() - self._started_at,
            "database": str(self._config.paths.database),
            "socket": str(self._config.paths.socket),
            "log_dir": str(self._config.paths.log_dir),
            "policy": self._config.scheduler.policy,
            "paused": self._scheduler.paused,
            "clients": len(self._sessions),
            "adapters": [
                encode_metadata_spec(self._registry.specs()[name]) for name in self._registry.names
            ],
            "rejected_adapters": [
                {"name": r.name, "source": r.source, "reason": r.reason}
                for r in self._registry.rejected
            ],
            "counts": {state.value: count for state, count in self._repo.count_by_state().items()},
        }

    def _daemon_shutdown(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        if self._on_shutdown:
            self._on_shutdown()
        return {"stopping": True}

    def _system_snapshot(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        return encode_snapshot(self._monitor.snapshot())

    def _scheduler_cpu_mode(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Switch -- or, with ``preview``, describe switching -- what a core means.

        The preview exists because the consequences are not obvious and some are not
        reversible by waiting: going from threads to cores on a busy machine can leave the
        ledger over-committed until jobs finish, and a queued job asking for more cores than
        the new total will never start. The dashboard shows both before asking to confirm.

        Persisted in the daemon's settings table rather than config.toml (migration 007).
        Choosing the config file's own value clears the override, so "back to what the file
        says" needs no special command.
        """
        current = self._resources.cpu_mode
        raw = str(params.get("mode") or "").strip().lower()
        if raw in ("", "toggle"):
            target = CpuMode.PHYSICAL if current is CpuMode.LOGICAL else CpuMode.LOGICAL
        else:
            try:
                target = CpuMode(raw)
            except ValueError as exc:
                valid = ", ".join(mode.value for mode in CpuMode)
                raise ValidationError(f"Unknown CPU mode {raw!r}. Valid modes: {valid}") from exc

        total, schedulable = self._resources.totals_for(target)
        allocated = self._resources.allocated_cores
        stranded = [
            {"id": job.id, "name": job.name, "cores": job.cores}
            for job in self._repo.queued()
            if job.cores > schedulable
        ]
        result: dict[str, Any] = {
            "previous": current.value,
            "mode": target.value,
            "previous_total": self._resources.total_cores,
            "total_cores": total,
            "schedulable_cores": schedulable,
            "allocated_cores": allocated,
            "over_committed": allocated > schedulable,
            "stranded": stranded,
            "relabel_only": bool(self._config.scheduler.total_cores),
            "applied": False,
        }
        if params.get("preview") or target is current:
            return result

        configured = self._config.scheduler.resolved_cpu_mode
        if target is configured:
            self._repo.clear_setting(CPU_MODE_SETTING)
            self._resources.set_cpu_mode(target, source="config")
        else:
            self._repo.set_setting(CPU_MODE_SETTING, target.value)
            self._resources.set_cpu_mode(target, source="interface")

        log.info(
            "CPU mode %s -> %s (%d -> %d %s)",
            current.value,
            target.value,
            result["previous_total"],
            total,
            "threads" if target is CpuMode.LOGICAL else "cores",
        )
        # More capacity may admit waiting work immediately; less changes nothing running.
        self._scheduler.nudge()
        self._bus.publish(Event.SYSTEM_STATS, encode_snapshot(self._monitor.snapshot()))
        result["applied"] = True
        return result

    # -- jobs --------------------------------------------------------------------------------------

    def _job_submit(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        """Inspect a case and queue it.

        Validation errors block submission unless ``force`` is set, because the user
        sometimes knows things the validator does not -- a mesh built by a step the adapter
        cannot see, a solver installed somewhere unusual.
        """
        workdir = Path(str(params.get("workdir", ""))).expanduser()
        cores = _as_int(params.get("cores"), default=1)
        ram_mb = params.get("ram_mb")
        force = bool(params.get("force"))
        # Built here, once, so the CPU/GPU consistency rules are applied to every
        # submission path -- the CLI, the wizard, and a hand-written socket client alike.
        request = ResourceRequest.build(
            cores=cores,
            ram_mb=int(ram_mb) if ram_mb else None,
            gpus=_as_optional_int(params.get("gpus")),
            resource=params.get("resource"),
        )
        # Optional, and absent from almost every submission. Resolved through the same
        # prefix expansion as every other job id the user types, so an unknown or
        # ambiguous one is refused here rather than becoming a job that waits forever.
        after = str(params.get("depends_on_job_id") or "").strip()
        depends_on_job_id = self._repo.resolve_id(after) if after else None

        start_from_latest = bool(params.get("start_from_latest"))
        result = self._inspector.inspect(
            workdir,
            cores=cores,
            ram_mb=request.ram_mb,
            gpus=request.gpus,
            start_from_latest=start_from_latest,
            solver=params.get("solver"),
            job_name=str(params.get("name") or ""),
            build_plan=False,
        )
        assert result.detection is not None

        if not result.report.passed and not force:
            raise ValidationError(
                "This case did not pass pre-flight validation: "
                f"{result.report.summary()}. Submit with force to queue it anyway.",
                detail={"validation": encode_report(result.report)},
            )

        metadata = result.metadata or CaseMetadata.empty(result.detection.solver)
        if result.detection.entry is not None:
            metadata = CaseMetadata(
                spec=metadata.spec,
                case=metadata.case,
                extra={**metadata.extra, "entry": str(result.detection.entry)},
            )

        spec = JobSpec(
            workdir=workdir,
            solver=result.detection.solver,
            solver_binary=result.detection.solver_binary,
            resources=request,
            name=str(params.get("name") or ""),
            priority=_as_int(params.get("priority"), default=0),
            tags=frozenset(params.get("tags") or ()),
            note=params.get("note"),
            metadata=metadata,
            depends_on_job_id=depends_on_job_id,
            start_from_latest=start_from_latest,
        )

        can, reason = self._resources.can_admit(spec.resources)
        # A request larger than the machine can ever satisfy is refused now rather than
        # queued forever. Checked for both pools: "20 cores on a 16-core box" and "2 GPUs
        # on a machine with one" fail for the same reason and deserve the same answer.
        impossible = (
            cores > self._resources.schedulable_cores
            or request.gpus > self._resources.schedulable_gpus
        )
        if not can and impossible:
            raise ValidationError(
                f"This job asks for {request.describe()}, which it can never get: {reason}",
                detail={
                    "schedulable_cores": self._resources.schedulable_cores,
                    "schedulable_gpus": self._resources.schedulable_gpus,
                },
            )

        job = self._repo.create(spec, state=JobState.QUEUED)
        # Log paths derive from the job id, which does not exist until the row does, so
        # they are assigned immediately afterwards rather than passed in.
        job = assign_log_paths(
            self._repo, self._config, job, getattr(result.adapter, "log_name", DEFAULT_LOG_NAME)
        )

        self._bus.publish(Event.JOB_STATE, encode_job(job))
        self._scheduler.nudge()
        log.info("Queued job %s (%s, %s)", job.id[:8], job.name, job.resources.describe())
        return {
            "job": encode_job(job, queue_position=self._repo.queue_positions().get(job.id)),
            "validation": encode_report(result.report),
        }

    def _job_list(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        states = params.get("states")
        parsed = [JobState(str(s).upper()) for s in states] if states else None
        page = self._repo.list_jobs(
            states=parsed,
            limit=_as_int(params.get("limit"), default=200),
            offset=_as_int(params.get("offset"), default=0),
        )
        return encode_page(page, self._repo.queue_positions())

    def _job_get(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        job_id = self._resolve(params)
        job = self._repo.get(job_id)
        return {
            "job": encode_job(job, queue_position=self._repo.queue_positions().get(job.id)),
            "events": [encode_event(e) for e in self._repo.events(job_id)],
            "notes": [encode_note(n) for n in self._repo.notes(job_id)],
            "samples": [encode_sample(s) for s in self._repo.samples(job_id, limit=200)],
            "waiting_because": self._scheduler.explain(job),
        }

    async def _job_cancel(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        job_id = self._resolve(params)
        job = self._repo.get(job_id)
        force = bool(params.get("force"))

        if job.is_terminal:
            raise ValidationError(f"Job {job.name} has already finished ({job.state.value})")

        if self._executor.is_running(job_id):
            await self._executor.cancel(job_id, force=force)
            return {"cancelling": True, "id": job_id}

        job = self._repo.transition(job_id, JobState.CANCELLED, detail="cancelled by the user")
        self._bus.publish(Event.JOB_STATE, encode_job(job))
        self._scheduler.nudge()
        return {"cancelling": False, "id": job_id, "job": encode_job(job)}

    async def _job_repartition(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Change a job's core count: at once if it is waiting, at its next write if running.

        A running job is never taken out of the scheduler's hands: it stays RUNNING while the
        solver finishes its step, then takes the ``RUNNING -> QUEUED`` edge and is admitted
        again like anything else. So a repartition cannot strand an allocation, and a job
        caught mid-resize is -- correctly -- either running or queued.

        A queued or held job has nothing to pause, so its count is simply changed. Its case is
        validated again on the new count first, as a submission would be, because the count
        is what decomposition and the validator's checks are measured against.
        """
        job_id = self._resolve(params)
        job = self._repo.get(job_id)
        cores = _as_int(params.get("cores"), default=0)

        request = ResourceRequest.build(
            cores=cores, gpus=job.gpus or None, resource=job.resource_kind.value
        )
        if request.cores > self._resources.schedulable_cores:
            raise ValidationError(
                f"{request.describe()} can never be scheduled on this machine "
                f"({self._resources.schedulable_cores} schedulable), so the job would "
                "stop and never resume."
            )
        if request.cores == job.cores:
            raise ValidationError(
                f"{job.name} is already set to {self._resources.describe_cores(job.cores)}"
            )

        if job.state in (JobState.QUEUED, JobState.HELD) and not self._executor.is_running(
            job_id
        ):
            return await self._recore_waiting(job, request.cores, force=bool(params.get("force")))

        if not self._executor.is_running(job_id):
            raise ValidationError(
                f"{job.name} is {job.state.value.lower()}, so its core count cannot change"
            )

        await self._executor.repartition(job_id, request.cores)
        updated = self._repo.get(job_id)
        return {
            "job": encode_job(updated),
            "pausing": True,
            "cores": request.cores,
        }

    async def _recore_waiting(self, job: Job, cores: int, *, force: bool) -> dict[str, Any]:
        entry = job.metadata.extra.get("entry")
        result = await asyncio.to_thread(
            self._inspector.inspect,
            job.workdir,
            cores=cores,
            ram_mb=job.ram_estimate_mb,
            gpus=job.gpus,
            solver=job.solver,
            entry=Path(str(entry)) if entry else None,
            job_name=job.name,
            build_plan=False,
            start_from_latest=job.start_from_latest,
        )
        if not result.report.passed and not force:
            raise ValidationError(
                f"On {self._resources.describe_cores(cores)} this case does not pass "
                f"validation: {result.report.summary()}. Pass force to change it anyway.",
                detail={"validation": encode_report(result.report)},
            )
        updated = self._repo.set_waiting_cores(job.id, cores)
        self._bus.publish(Event.JOB_STATE, encode_job(updated))
        # Fewer cores may let it start now; more may let something behind it go first.
        self._scheduler.nudge()
        log.info("Job %s re-cored to %d before starting", job.id[:8], cores)
        return {
            "job": encode_job(updated),
            "pausing": False,
            "cores": cores,
            "validation": encode_report(result.report),
        }

    def _job_hold(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        job = self._scheduler.hold(self._resolve(params))
        self._bus.publish(Event.JOB_STATE, encode_job(job))
        return encode_job(job)

    def _job_release(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        job = self._scheduler.release(self._resolve(params))
        self._bus.publish(Event.JOB_STATE, encode_job(job))
        return encode_job(job)

    def _job_priority(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        job = self._scheduler.set_priority(
            self._resolve(params), _as_int(params.get("priority"), default=0)
        )
        self._bus.publish(Event.JOB_STATE, encode_job(job))
        return encode_job(job)

    def _job_delete(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        job_id = self._resolve(params)
        job = self._repo.get(job_id)
        self._repo.delete(job_id)
        if params.get("purge_logs"):
            _purge(self._config.paths.job_dir(job_id))
        self._bus.publish(Event.QUEUE_CHANGED, {"deleted": job_id})
        return {"deleted": job_id, "name": job.name}

    def _job_note(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        note = self._repo.add_note(self._resolve(params), str(params.get("body", "")))
        return encode_note(note)

    def _job_tag(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        job = self._repo.edit_tags(
            self._resolve(params),
            add=params.get("add") or (),
            remove=params.get("remove") or (),
        )
        self._bus.publish(Event.JOB_STATE, encode_job(job))
        return encode_job(job)

    def _job_provenance(self, session: ClientSession, params: dict[str, Any]) -> Any:
        record = self._repo.provenance(self._resolve(params))
        return encode_provenance(record) if record else None

    def _tags_list(self, session: ClientSession, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"name": name, "count": count} for name, count in self._repo.all_tags()]

    def _history_search(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        query = parse_query(str(params.get("query", "")), now=self._clock.now())
        page = self._repo.search(
            query,
            limit=_as_int(params.get("limit"), default=200),
            offset=_as_int(params.get("offset"), default=0),
        )
        return encode_page(page, self._repo.queue_positions())

    # -- cases -------------------------------------------------------------------------------------

    def _case_detect(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(params.get("path", ""))).expanduser()
        detections = self._inspector.detect(path)
        return {
            "path": str(path),
            "detections": [encode_detection(d) for d in detections],
        }

    def _case_validate(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        result = self._inspector.inspect(
            Path(str(params.get("path", ""))).expanduser(),
            cores=_as_int(params.get("cores"), default=1),
            gpus=_as_int(params.get("gpus"), default=0),
            solver=params.get("solver"),
            build_plan=False,
        )
        assert result.detection is not None
        return {
            "solver": result.detection.solver,
            "solver_binary": result.detection.solver_binary,
            "validation": encode_report(result.report),
            "metadata": result.metadata.to_json() if result.metadata else None,
        }

    async def _case_info(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """A full description of a case directory, for the information view (§9.8).

        The solver-specific reading is the adapter's -- ``describe_case`` -- and two sections
        are added here because they are Dispatch's own knowledge rather than the solver's:
        what version control says about the directory, and what Dispatch has already run in
        it. Neither is something an adapter should be shelling out to ``git`` to discover.

        Takes a path rather than a job id, so the view works on a directory being browsed
        before anything has been submitted, which is when it is most useful. A job id is
        accepted too and resolves to that job's working directory.
        """
        raw = params.get("path") or params.get("workdir")
        if raw:
            workdir = Path(str(raw)).expanduser()
        else:
            workdir = self._repo.get(self._resolve(params)).workdir

        try:
            resolved = workdir.resolve()
        except OSError as exc:
            raise ValidationError(f"Cannot read {workdir}: {exc}") from exc
        if not resolved.is_dir():
            raise ValidationError(f"{resolved} is not a directory")

        detection = self._registry.best_detection(resolved)
        if detection is None:
            raise ValidationError(
                f"No solver recognises {resolved}, so there is nothing to describe. "
                "Browse into a case directory."
            )

        adapter = self._registry.get(detection.solver)
        ctx = self._registry.context(
            resolved,
            entry=detection.entry,
            env=dict(os.environ),
            adapter=detection.solver,
            cpu_mode=self._resources.cpu_mode.value,
        )
        try:
            report = await asyncio.to_thread(adapter.describe_case, ctx)
        except Exception:
            log.exception("Adapter %s failed describing %s", detection.solver, resolved)
            raise ValidationError(
                f"The {detection.solver} adapter could not describe this case."
            ) from None

        sections = encode_case_report(report)
        sections["sections"].extend(
            [
                await self._version_control_section(resolved),
                self._dispatch_section(resolved),
            ]
        )
        sections["sections"] = [item for item in sections["sections"] if item]
        sections["path"] = str(resolved)
        return sections

    async def _version_control_section(self, workdir: Path) -> dict[str, Any]:
        """What git says about the case directory, reusing the provenance collector's probe.

        The same code that records provenance at job start, so the information view and the
        history cannot disagree about what the repository looked like. ``dirty`` leads,
        because it is the field that actually matters later: a clean commit recorded against a
        modified tree is a lie shaped like provenance.
        """
        git = await self._provenance_git(workdir)
        if git is None or not git.is_present:
            return {}
        fields = [
            {
                "label": "Commit",
                "value": (git.commit or "")[:12],
                "note": "uncommitted changes" if git.dirty else "clean",
                "important": git.dirty,
            }
        ]
        if git.branch:
            fields.append(
                {"label": "Branch", "value": git.branch, "note": "", "important": False}
            )
        if git.remote:
            fields.append(
                {"label": "Remote", "value": git.remote, "note": "", "important": False}
            )
        return {"title": "version control", "missing": "", "fields": fields}

    async def _provenance_git(self, workdir: Path) -> Any:
        """Probe git, never fatally. A missing git is a blank section, not an error."""
        try:
            return await self._executor.provenance_git(workdir)
        except Exception as exc:
            log.debug("Could not read git state of %s: %s", workdir, exc)
            return None

    def _dispatch_section(self, workdir: Path) -> dict[str, Any]:
        """What Dispatch has already run in this directory.

        Often the most useful thing on the page: "this case ran for six hours last Tuesday
        and failed" is not in any of the solver's own files.
        """
        jobs = self._repo.jobs_for_workdir(workdir)
        if not jobs:
            return {}

        fields: list[dict[str, Any]] = [
            {"label": "Runs", "value": str(len(jobs)), "note": "", "important": False}
        ]
        newest = jobs[0]
        fields.append(
            {
                "label": "Latest",
                "value": newest.state.value.lower(),
                "note": f"{newest.id[:8]} · {newest.resources.describe()}",
                "important": True,
            }
        )
        if newest.metrics.runtime_s:
            fields.append(
                {
                    "label": "Ran for",
                    "value": _duration(newest.metrics.runtime_s),
                    "note": "",
                    "important": False,
                }
            )
        if newest.exit_detail:
            fields.append(
                {
                    "label": "Failed because",
                    "value": newest.exit_detail.splitlines()[0],
                    "note": "",
                    "important": True,
                }
            )
        return {"title": "dispatch history", "missing": "", "fields": fields}

    def _render_target(self, params: dict[str, Any]) -> tuple[Path, Any, Any]:
        """The case directory a render request names, with its adapter and context.

        ``path`` for a directory being browsed, ``id`` for a job's working directory -- the
        same two ways the information view is opened.
        """
        raw = params.get("path") or params.get("workdir")
        if raw:
            workdir = Path(str(raw)).expanduser()
        else:
            workdir = self._repo.get(self._resolve(params)).workdir
        try:
            resolved = workdir.resolve()
        except OSError as exc:
            raise ValidationError(f"Cannot read {workdir}: {exc}") from exc
        if not resolved.is_dir():
            raise ValidationError(f"{resolved} is not a directory")

        detection = self._registry.best_detection(resolved)
        if detection is None:
            raise ValidationError(f"No solver recognises {resolved}, so there is nothing to render")
        adapter = self._registry.get(detection.solver)
        ctx = replace(
            self._registry.context(
                resolved,
                entry=detection.entry,
                env=dict(os.environ),
                adapter=detection.solver,
                cpu_mode=self._resources.cpu_mode.value,
            ),
            # A preview describes the render; it must not leave a generated script and a
            # reader stub behind in a case it was only asked about.
            dry_run=bool(params.get("dry_run")),
        )
        return resolved, adapter, ctx

    async def _case_fields(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """What a render of a case can be coloured by -- velocity, pressure, whatever it has.

        Read by the adapter from the case's own files, so the interface offers exactly this
        case's fields rather than a fixed list that is wrong for half the cases it is shown on.
        """
        resolved, adapter, ctx = self._render_target(params)
        fields = await asyncio.to_thread(adapter.visual_fields, ctx)
        return {
            "path": str(resolved),
            "fields": [
                {"name": item.name, "kind": item.kind, "label": item.display}
                for item in fields
            ],
        }

    async def _case_render(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Render a picture or a video of a case, with the adapter deciding how (§8.10).

        The adapter returns commands; :class:`~dispatch.daemon.renders.RenderManager` runs
        them. Started in the background and answered at once, unless ``wait`` is set: this
        daemon answers each client's requests in turn, so an animation run inside the request
        would freeze that interface for hours. Progress and the result arrive as ``renders``
        events. The CLI waits, because a command line has nothing else to do.

        Not a job, on purpose: a render holds no allocation, has no place in the history, and
        must not queue behind a week-long run.
        """
        resolved, adapter, ctx = self._render_target(params)
        try:
            kind = VisualKind(str(params.get("kind") or VisualKind.MESH.value).lower())
        except ValueError as exc:
            valid = ", ".join(item.value for item in VisualKind)
            raise ValidationError(f"Unknown render kind. Valid kinds: {valid}") from exc

        request = VisualRequest(
            kind=kind,
            preset=str(params.get("preset") or "isometric"),
            field=params.get("field") or None,
            frames=_as_optional_int(params.get("frames")),
            width=_as_int(params.get("width"), default=1600),
            height=_as_int(params.get("height"), default=1000),
            fps=_as_int(params.get("fps"), default=24),
            keep_frames=params.get("keep_frames", True) is not False,
        )
        plan = await asyncio.to_thread(adapter.visualise, ctx, request)
        if plan is None:
            tool = getattr(adapter, "renderer", "") or "A renderer for this solver"
            raise ValidationError(
                f"{tool} was not found on this machine, so Dispatch cannot render this case."
            )

        result = _render_result(plan, started=False, outputs=[])
        if ctx.dry_run:
            return result
        if params.get("wait"):
            produced = await self._renders.run_inline(resolved, kind.value, plan)
            return {**result, "rendered": True, "produced": produced}

        entry = self._renders.start(resolved, kind.value, plan)
        return {**result, "render_id": entry.id, "started": True}

    def _render_list(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        """Renders currently running, so a reopened interface can pick their progress back up."""
        return {"renders": [entry.describe() for entry in self._renders.active]}

    def _render_cancel(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        """Stop a render. Frames already written stay where they are."""
        render_id = str(params.get("id") or "").strip()
        if not render_id or not self._renders.cancel(render_id):
            raise ValidationError(f"No running render matches {render_id!r}")
        return {"cancelled": True, "id": render_id}

    def _case_dryrun(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        ram = params.get("ram_mb")
        report = self._inspector.dry_run(
            Path(str(params.get("workdir") or params.get("path", ""))).expanduser(),
            cores=_as_int(params.get("cores"), default=1),
            ram_mb=int(ram) if ram else None,
            gpus=_as_int(params.get("gpus"), default=0),
            resource=params.get("resource"),
            start_from_latest=bool(params.get("start_from_latest")),
            solver=params.get("solver"),
            job_name=str(params.get("name") or ""),
        )
        return encode_dry_run(report)

    async def _job_series(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Read a job's log and return what its adapter found worth plotting.

        Off the event loop: parsing tens of megabytes of residuals is real work, and the
        scheduler must not stop while somebody looks at a chart. This is also why the
        method exists at all rather than the interface parsing the file itself -- the
        parser belongs to the adapter, and the interface is not allowed to know adapters
        exist (§3).
        """
        job = self._repo.get(self._resolve(params))
        datasets = await asyncio.to_thread(
            extract_datasets, job, registry=self._registry, config=self._config.plot
        )
        # The log dataset is also returned flat, as it always was. A client from before
        # datasets existed keeps working, and the handshake does not have to refuse it over
        # a purely additive field.
        from_log = next((item for item in datasets if item.key == "log"), None)
        return {
            "id": job.id,
            "name": job.name,
            "solver": job.solver,
            "path": str(job.output_path) if job.output_path else None,
            "datasets": [encode_dataset(item) for item in datasets],
            **encode_plot_data(from_log.data if from_log else PlotData()),
        }

    async def _job_metrics(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """The latest values a case's own output files hold, for a one-line summary.

        Deliberately not ``job.series`` with the last point taken: this reads only the case's
        post-processing files and never the log, so the dashboard can ask it about a job
        without paying to parse a gigabyte of residuals.
        """
        job = self._repo.get(self._resolve(params))
        latest = await asyncio.to_thread(
            latest_case_values, job, registry=self._registry
        )
        return {"id": job.id, "name": job.name, "datasets": latest}

    async def _projects_search(
        self, session: ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Find project directories by name under the configured root.

        Daemon-side for the same reason ``fs.list`` is: the results carry the "this looks
        like a case" marks, and those come from the real adapters. Off the event loop
        because a cold projects tree is a few thousand ``getdents`` calls.
        """
        query = str(params.get("query") or "").strip()
        limit = _as_int(params.get("limit"), default=self._config.projects.limit)
        result = await asyncio.to_thread(
            search_projects,
            self._config.projects,
            query,
            limit=max(1, limit),
            # The cheap, stat-only screen -- not `detect`, which parses and would cost
            # half a millisecond per directory on a tree of thousands.
            is_case=self._registry.looks_like_case,
        )

        detect = bool(params.get("detect", True))
        entries: list[dict[str, Any]] = []
        for hit in result.hits:
            encoded = encode_project_hit(hit)
            detection = None
            if detect:
                # Only for the results actually being shown: detection is cheap per
                # directory and ruinous across a whole tree.
                with contextlib.suppress(Exception):
                    detection = self._registry.best_detection(hit.path)
            encoded["case"] = detection.solver if detection else None
            encoded["label"] = detection.label if detection else None
            entries.append(encoded)

        return {
            "root": str(result.root),
            "query": result.query,
            "results": entries,
            "scanned": result.scanned,
            "truncated": result.truncated,
            "error": result.error,
        }

    def _sweep_submit(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        """Queue a directory of same-solver cases as one sweep.

        The cases are re-detected here rather than taken from the client, so a sweep is
        always built from what the daemon can currently see on disk -- the same rule that
        makes ``fs.list`` authoritative about what is a case.

        Every member is an ordinary job. The sweep contributes the per-job core count and
        the concurrency cap, and nothing else: no member is special, none of them is a
        parent, and each is admitted, supervised, cancelled and recorded exactly like a
        job submitted on its own.
        """
        root = Path(str(params.get("root") or params.get("workdir") or "")).expanduser()
        cores = _as_int(params.get("cores_per_job"), default=1)
        concurrency = _as_int(params.get("concurrency"), default=1)

        detection = self._registry.detect_sweep(root)
        if detection is None:
            raise ValidationError(
                f"{root} is not a sweep folder: a sweep is a directory whose "
                "subdirectories are all cases of the same solver."
            )

        request = ResourceRequest.build(
            cores=cores,
            ram_mb=_as_optional_int(params.get("ram_mb")),
            gpus=_as_optional_int(params.get("gpus")),
            resource=params.get("resource"),
        )
        # Refused now rather than queued forever, exactly as a single oversized job is.
        # For a sweep this matters more, not less: without it a mistyped core count queues
        # forty jobs that can never start instead of one.
        if (
            request.cores > self._resources.schedulable_cores
            or request.gpus > self._resources.schedulable_gpus
        ):
            raise ValidationError(
                f"Each job in this sweep asks for {request.describe()}, which it can never "
                f"get on this machine ({self._resources.schedulable_cores} cores "
                f"schedulable)."
            )

        spec = SweepSpec(
            root=root,
            solver=detection.solver,
            cases=detection.cases,
            cores_per_job=request.cores,
            concurrency=concurrency,
            name=str(params.get("name") or ""),
            start_from_latest=bool(params.get("start_from_latest")),
        )

        members: list[JobSpec] = []
        for position, case in enumerate(spec.cases):
            result = self._inspector.inspect(
                case,
                cores=request.cores,
                ram_mb=request.ram_mb,
                gpus=request.gpus,
                solver=detection.solver,
                build_plan=False,
                start_from_latest=spec.start_from_latest,
            )
            metadata = result.metadata or CaseMetadata.empty(detection.solver)
            if result.detection is not None and result.detection.entry is not None:
                metadata = CaseMetadata(
                    spec=metadata.spec,
                    case=metadata.case,
                    extra={**metadata.extra, "entry": str(result.detection.entry)},
                )
            members.append(
                JobSpec(
                    workdir=case,
                    solver=detection.solver,
                    solver_binary=(
                        result.detection.solver_binary if result.detection is not None else None
                    ),
                    resources=request,
                    name=case.name,
                    priority=_as_int(params.get("priority"), default=0),
                    tags=frozenset(params.get("tags") or ()),
                    metadata=metadata,
                    sweep_id=spec.sweep_id,
                    sweep_position=position,
                    start_from_latest=spec.start_from_latest,
                )
            )

        sweep, jobs = self._repo.create_sweep(spec, members)
        jobs = [
            assign_log_paths(self._repo, self._config, job, DEFAULT_LOG_NAME) for job in jobs
        ]

        for job in jobs:
            self._bus.publish(Event.JOB_STATE, encode_job(job))
        self._scheduler.nudge()
        log.info(
            "Queued sweep %s (%s, %d cases, %d cores each, %d at a time)",
            sweep.id[:8],
            sweep.name,
            len(jobs),
            sweep.cores_per_job,
            sweep.concurrency,
        )
        positions = self._repo.queue_positions()
        return {
            "sweep": encode_sweep(sweep),
            "jobs": [encode_job(job, queue_position=positions.get(job.id)) for job in jobs],
        }

    def _sweep_list(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        """Every sweep with its live member counts, for the queue view."""
        return {"sweeps": [encode_sweep(sweep) for sweep in self._repo.sweeps()]}

    def _fs_list(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        """List subdirectories, marking those that look like cases.

        Runs daemon-side so the "this directory is a case" hint comes from the real
        adapters. The TUI stays solver-ignorant, which is the whole point of §3's layering.
        """
        raw = str(params.get("path") or Path.home())
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except OSError as exc:
            raise ValidationError(f"Cannot read {path}: {exc}") from exc
        if not resolved.is_dir():
            raise ValidationError(f"{resolved} is not a directory")

        entries: list[dict[str, Any]] = []
        try:
            children = sorted(
                (child for child in resolved.iterdir() if child.is_dir()),
                key=lambda child: child.name.lower(),
            )
        except PermissionError as exc:
            raise ValidationError(f"Cannot read {resolved}: permission denied") from exc

        detect = bool(params.get("detect", True))
        for child in children:
            if child.name.startswith(".") and not params.get("hidden"):
                continue
            detection = None
            if detect:
                with contextlib.suppress(Exception):
                    detection = self._registry.best_detection(child)
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "case": detection.solver if detection else None,
                    "label": detection.label if detection else None,
                }
            )

        here = self._registry.best_detection(resolved) if detect else None
        # Only asked when the directory is not itself a case, which is also the first thing
        # detect_sweep checks. The answer rides along with the listing the browser already
        # makes, so noticing a sweep costs the TUI no extra round trip.
        sweep = self._registry.detect_sweep(resolved) if detect and here is None else None
        return {
            "path": str(resolved),
            "parent": str(resolved.parent) if resolved.parent != resolved else None,
            "entries": entries,
            "case": here.solver if here else None,
            "sweep": (
                {
                    "solver": sweep.solver,
                    "count": sweep.count,
                    "cases": [c.name for c in sweep.cases],
                }
                if sweep is not None
                else None
            ),
        }

    # -- subscriptions -----------------------------------------------------------------------------

    def _subscribe(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        topics = params.get("topics") or []
        self._bus.set_topics(session.subscription, [str(t) for t in topics])
        return {"topics": sorted(session.subscription.topics)}

    def _unsubscribe(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        self._bus.set_topics(session.subscription, [])
        return {"topics": []}

    # -- helpers -----------------------------------------------------------------------------------

    def _resolve(self, params: dict[str, Any]) -> str:
        """Expand a possibly-abbreviated job id.

        Users type the first few characters of a UUID; requiring all 36 would make the CLI
        unusable, and guessing between two matches would be worse than either.
        """
        raw = str(params.get("id", "")).strip()
        if not raw:
            raise ValidationError("A job id is required")
        return self._repo.resolve_id(raw)


def _render_result(plan: Any, *, started: bool, outputs: Sequence[str]) -> dict[str, Any]:
    """What a render request reports: the plan, and what it produced if it ran."""
    return {
        "tool": plan.tool,
        "rendered": started,
        "notes": list(plan.notes),
        "outputs": [str(path) for path in plan.outputs],
        "produced": list(outputs),
        "command": list(plan.steps[0].argv) if plan.steps else [],
        "steps": [
            {"description": step.description, "command": list(step.argv)}
            for step in plan.steps
        ],
    }


def _duration(seconds: float | None) -> str:
    """Render a runtime compactly: ``3d 4h``, ``2h 14m``, ``45s``.

    A copy of the CLI's, deliberately: the daemon may not import the CLI, and three lines of
    arithmetic is a far smaller price than a shared module that exists only to hold it.
    """
    if not seconds or seconds < 0:
        return "-"
    total = int(seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    """Distinguish "the user said zero" from "the user did not say".

    ``--gpus 0`` is a statement that this is CPU work; omitting the flag entirely leaves
    the question to ``--resource``, and the two must not collapse into the same value.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _purge(directory: Path) -> None:
    """Delete a job's log directory, best effort."""
    import shutil

    with contextlib.suppress(OSError):
        shutil.rmtree(directory)


def topics_of(subscriptions: Sequence[Subscription]) -> set[str]:
    """Union of topics across subscriptions. Used by the demand-driven sampler."""
    active: set[str] = set()
    for subscription in subscriptions:
        active |= subscription.topics
    return active
