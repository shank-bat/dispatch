"""The daemon/client wire protocol: message shapes, method names, and serialisation.

Newline-delimited JSON over an ``AF_UNIX`` stream. Chosen so that the entire protocol can
be driven by hand from an SSH session::

    socat - UNIX-CONNECT:~/.local/run/dispatch/daemon.sock
    {"t":"req","id":1,"method":"job.list","params":{}}

On a headless machine that debuggability is worth more than the microseconds a binary
format would save on a protocol that carries maybe ten messages a second.

See ``docs/ARCHITECTURE.md`` §7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from dispatch.core.metadata import MetadataSpec
from dispatch.core.models import Job, JobEvent, Note, Page, Sample, SystemSnapshot
from dispatch.core.plan import DryRunReport, ExecutionPlan
from dispatch.core.provenance import Provenance
from dispatch.core.validation import ValidationReport

__all__ = [
    "PROTOCOL_VERSION",
    "Event",
    "Method",
    "Request",
    "Response",
    "Topic",
    "encode_job",
    "encode_snapshot",
]

PROTOCOL_VERSION: Final = 1
"""Bumped on any incompatible change to messages or method semantics.

A mismatch is refused at the handshake with a message naming both versions. The realistic
skew is a client from a fresh install talking to a daemon nobody restarted after an
upgrade, so the error must say "restart the daemon", not raise ``KeyError``.
"""

MAX_MESSAGE_BYTES: Final = 4 * 1024 * 1024
"""Ceiling on a single framed message, enforced in both directions."""


class Method(StrEnum):
    """Every request the daemon answers."""

    HELLO = "hello"
    DAEMON_INFO = "daemon.info"
    DAEMON_SHUTDOWN = "daemon.shutdown"
    SYSTEM_SNAPSHOT = "system.snapshot"

    JOB_SUBMIT = "job.submit"
    JOB_LIST = "job.list"
    JOB_GET = "job.get"
    JOB_CANCEL = "job.cancel"
    JOB_HOLD = "job.hold"
    JOB_RELEASE = "job.release"
    JOB_PRIORITY = "job.priority"
    JOB_DELETE = "job.delete"
    JOB_NOTE = "job.note"
    JOB_TAG = "job.tag"
    JOB_PROVENANCE = "job.provenance"

    TAGS_LIST = "tags.list"
    HISTORY_SEARCH = "history.search"

    CASE_DETECT = "case.detect"
    CASE_VALIDATE = "case.validate"
    CASE_DRYRUN = "case.dryrun"
    FS_LIST = "fs.list"

    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


class Topic(StrEnum):
    """Event streams a client may subscribe to.

    Subscription is per-topic because sampling is driven by demand: with nobody watching
    ``system``, the daemon does not measure the machine at all.
    """

    JOBS = "jobs"
    QUEUE = "queue"
    SYSTEM = "system"
    DAEMON = "daemon"


class Event(StrEnum):
    """Server-initiated notifications."""

    JOB_STATE = "job.state"
    JOB_PROGRESS = "job.progress"
    QUEUE_CHANGED = "queue.changed"
    SYSTEM_STATS = "system.stats"
    DAEMON_SHUTDOWN = "daemon.shutdown"
    RESYNC = "resync"
    """The client's event queue overflowed and was dropped; refetch everything."""


EVENT_TOPICS: Final[dict[Event, Topic]] = {
    Event.JOB_STATE: Topic.JOBS,
    Event.JOB_PROGRESS: Topic.JOBS,
    Event.QUEUE_CHANGED: Topic.QUEUE,
    Event.SYSTEM_STATS: Topic.SYSTEM,
    Event.DAEMON_SHUTDOWN: Topic.DAEMON,
    Event.RESYNC: Topic.DAEMON,
}


@dataclass(frozen=True, slots=True)
class Request:
    """A client-to-daemon call."""

    id: int
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"t": "req", "id": self.id, "method": self.method, "params": self.params}


@dataclass(frozen=True, slots=True)
class Response:
    """A daemon-to-client reply, matched to a request by ``id``."""

    id: int
    ok: bool
    result: Any = None
    error: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"t": "res", "id": self.id, "ok": self.ok}
        if self.ok:
            payload["result"] = self.result
        else:
            payload["error"] = self.error or {}
        return payload

    @classmethod
    def failure(cls, request_id: int, code: str, message: str, **detail: Any) -> Response:
        """Build an error response carrying a stable machine-readable ``code``."""
        return cls(
            id=request_id,
            ok=False,
            error={"code": code, "message": message, "detail": detail},
        )


@dataclass(frozen=True, slots=True)
class Notification:
    """A server-pushed event. Has no ``id``: nothing replies to it."""

    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"t": "evt", "event": self.event, "data": self.data}


# -- encoders -------------------------------------------------------------------------------
#
# Domain objects are converted here rather than given `to_json` methods, so that `core`
# stays free of any knowledge that a wire protocol exists.


def encode_job(job: Job, *, queue_position: int | None = None) -> dict[str, Any]:
    """Render a job for the wire.

    ``queue_position`` is passed in rather than read off the job because it is derived
    from the whole queue, not stored on the row (§13.3).
    """
    return {
        "id": job.id,
        "seq": job.seq,
        "name": job.name,
        "workdir": str(job.workdir),
        "solver": job.solver,
        "solver_binary": job.solver_binary,
        "cores": job.cores,
        "ram_estimate_mb": job.ram_estimate_mb,
        "priority": job.priority,
        "state": job.state.value,
        "queue_position": queue_position,
        "depends_on_job_id": job.depends_on_job_id,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "exit_code": job.exit_code,
        "exit_reason": job.exit_reason.value if job.exit_reason else None,
        "exit_signal": job.exit_signal,
        "exit_detail": job.exit_detail,
        "stdout_path": str(job.stdout_path) if job.stdout_path else None,
        "stderr_path": str(job.stderr_path) if job.stderr_path else None,
        "pid": job.pid,
        "tags": sorted(job.tags),
        "metadata": job.metadata.to_json(),
        "metrics": {
            "peak_rss_mb": job.metrics.peak_rss_mb,
            "mean_cpu_pct": job.metrics.mean_cpu_pct,
            "runtime_s": job.metrics.runtime_s,
        },
    }


def encode_snapshot(snapshot: SystemSnapshot) -> dict[str, Any]:
    """Render a system snapshot for the wire."""
    return {
        "timestamp": snapshot.timestamp,
        "hostname": snapshot.hostname,
        "total_cores": snapshot.total_cores,
        "allocated_cores": snapshot.allocated_cores,
        "reserved_cores": snapshot.reserved_cores,
        "free_cores": snapshot.free_cores,
        "cpu_percent": snapshot.cpu_percent,
        "per_core_percent": list(snapshot.per_core_percent),
        "total_ram_mb": snapshot.total_ram_mb,
        "used_ram_mb": snapshot.used_ram_mb,
        "available_ram_mb": snapshot.available_ram_mb,
        "ram_percent": snapshot.ram_percent,
        "load_average": list(snapshot.load_average),
        "uptime_s": snapshot.uptime_s,
    }


def encode_event(event: JobEvent) -> dict[str, Any]:
    """Render an audit-trail entry."""
    return {"id": event.id, "ts": event.ts, "kind": event.kind, "detail": event.detail}


def encode_note(note: Note) -> dict[str, Any]:
    """Render a note."""
    return {"id": note.id, "ts": note.ts, "body": note.body}


def encode_sample(sample: Sample) -> dict[str, Any]:
    """Render a resource sample."""
    return {"ts": sample.ts, "rss_mb": sample.rss_mb, "cpu_pct": sample.cpu_pct}


def encode_report(report: ValidationReport) -> dict[str, Any]:
    """Render a validation report."""
    return {
        "passed": report.passed,
        "summary": report.summary(),
        "worst": report.worst.label if report.worst else None,
        "findings": [
            {
                "severity": finding.severity.label,
                "message": finding.message,
                "hint": finding.hint,
                "path": finding.path,
                "code": finding.code,
            }
            for finding in report.findings
        ],
    }


def encode_plan(plan: ExecutionPlan) -> dict[str, Any]:
    """Render an execution plan, including the exact commands."""
    return {
        "steps": [
            {
                "kind": step.kind.value,
                "description": step.description,
                "argv": list(step.argv),
                "command": step.render(),
                "cwd": str(step.cwd),
                "on_failure": step.on_failure.value,
                "timeout_s": step.timeout_s,
            }
            for step in plan.steps
        ]
    }


def encode_detection(detection: Any) -> dict[str, Any]:
    """Render a solver detection."""
    return {
        "solver": detection.solver,
        "confidence": detection.confidence,
        "solver_binary": detection.solver_binary,
        "label": detection.label,
        "entry": str(detection.entry) if detection.entry else None,
        "detail": dict(detection.detail),
    }


def encode_dry_run(report: DryRunReport) -> dict[str, Any]:
    """Render a dry-run report."""
    return {
        "workdir": str(report.workdir),
        "solver": report.solver,
        "solver_binary": report.solver_binary,
        "cores": report.cores,
        "would_submit": report.would_submit,
        "detections": [encode_detection(d) for d in report.detections],
        "validation": encode_report(report.validation),
        "plan": encode_plan(report.plan) if report.plan else None,
        "suggested_tags": list(report.suggested_tags),
        "projection": (
            {
                "cores_requested": report.projection.cores_requested,
                "cores_free": report.projection.cores_free,
                "cores_total": report.projection.cores_total,
                "would_start_immediately": report.projection.would_start_immediately,
                "blocking_reason": report.projection.blocking_reason,
            }
            if report.projection
            else None
        ),
    }


def encode_provenance(prov: Provenance) -> dict[str, Any]:
    """Render a reproducibility record."""
    return prov.to_json()


def encode_page(page: Page[Job], positions: dict[str, int] | None = None) -> dict[str, Any]:
    """Render a page of jobs."""
    lookup = positions or {}
    return {
        "items": [encode_job(job, queue_position=lookup.get(job.id)) for job in page.items],
        "total": page.total,
        "offset": page.offset,
        "has_more": page.has_more,
    }


def encode_metadata_spec(spec: MetadataSpec) -> dict[str, Any]:
    """Render an adapter's metadata spec, so the TUI can label fields it has never seen."""
    return {
        "adapter": spec.ref.adapter,
        "version": spec.ref.version,
        "fields": [
            {
                "key": item.key,
                "type": item.type.value,
                "label": item.label,
                "unit": item.unit,
                "display_order": item.display_order,
            }
            for item in spec.ordered
        ],
    }
