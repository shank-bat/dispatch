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
from pathlib import Path
from typing import Any

from dispatch.adapters.registry import AdapterRegistry
from dispatch.core.clock import Clock, SystemClock
from dispatch.core.config import Config
from dispatch.core.errors import DispatchError, ValidationError
from dispatch.core.metadata import CaseMetadata
from dispatch.core.models import JobSpec, ResourceRequest
from dispatch.core.query import parse_query
from dispatch.core.states import JobState
from dispatch.daemon.dryrun import CaseInspector
from dispatch.daemon.events import EventBus, Subscription
from dispatch.daemon.executor import JobExecutor
from dispatch.daemon.monitor import SystemMonitor
from dispatch.daemon.resources import ResourceModel
from dispatch.daemon.scheduler import Scheduler
from dispatch.db.repository import JobRepository
from dispatch.ipc.codec import ProtocolError, read_message, write_message
from dispatch.ipc.protocol import (
    PROTOCOL_VERSION,
    Event,
    Method,
    Response,
    encode_detection,
    encode_dry_run,
    encode_event,
    encode_job,
    encode_metadata_spec,
    encode_note,
    encode_page,
    encode_provenance,
    encode_report,
    encode_sample,
    encode_snapshot,
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
        """Tell clients the daemon is going away, then close the socket."""
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
            Method.JOB_SUBMIT: self._job_submit,
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
            Method.TAGS_LIST: self._tags_list,
            Method.HISTORY_SEARCH: self._history_search,
            Method.CASE_DETECT: self._case_detect,
            Method.CASE_VALIDATE: self._case_validate,
            Method.CASE_DRYRUN: self._case_dryrun,
            Method.FS_LIST: self._fs_list,
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
        # Optional, and absent from almost every submission. Resolved through the same
        # prefix expansion as every other job id the user types, so an unknown or
        # ambiguous one is refused here rather than becoming a job that waits forever.
        after = str(params.get("depends_on_job_id") or "").strip()
        depends_on_job_id = self._repo.resolve_id(after) if after else None

        result = self._inspector.inspect(
            workdir,
            cores=cores,
            ram_mb=int(ram_mb) if ram_mb else None,
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
            resources=ResourceRequest(cores=cores, ram_mb=int(ram_mb) if ram_mb else None),
            name=str(params.get("name") or ""),
            priority=_as_int(params.get("priority"), default=0),
            tags=frozenset(params.get("tags") or ()),
            note=params.get("note"),
            metadata=metadata,
            depends_on_job_id=depends_on_job_id,
        )

        can, reason = self._resources.can_admit(spec.resources)
        if not can and cores > self._resources.schedulable_cores:
            raise ValidationError(
                f"This job asks for {cores} cores, which it can never get: {reason}",
                detail={"schedulable_cores": self._resources.schedulable_cores},
            )

        job = self._repo.create(spec, state=JobState.QUEUED)
        # Log paths derive from the job id, which does not exist until the row does, so
        # they are assigned immediately afterwards rather than passed in.
        log_dir = self._config.paths.job_dir(job.id)
        log_dir.mkdir(parents=True, exist_ok=True)
        job = self._repo.set_log_paths(
            job.id, stdout=log_dir / "stdout.log", stderr=log_dir / "stderr.log"
        )

        self._bus.publish(Event.JOB_STATE, encode_job(job))
        self._scheduler.nudge()
        log.info("Queued job %s (%s, %d cores)", job.id[:8], job.name, job.cores)
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

    def _case_dryrun(self, session: ClientSession, params: dict[str, Any]) -> dict[str, Any]:
        ram = params.get("ram_mb")
        report = self._inspector.dry_run(
            Path(str(params.get("workdir") or params.get("path", ""))).expanduser(),
            cores=_as_int(params.get("cores"), default=1),
            ram_mb=int(ram) if ram else None,
            solver=params.get("solver"),
            job_name=str(params.get("name") or ""),
        )
        return encode_dry_run(report)

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
        return {
            "path": str(resolved),
            "parent": str(resolved.parent) if resolved.parent != resolved else None,
            "entries": entries,
            "case": here.solver if here else None,
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


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
