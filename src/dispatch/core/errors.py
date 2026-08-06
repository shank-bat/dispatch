"""Exception hierarchy for Dispatch.

Every exception carries a stable :attr:`DispatchError.code`. The IPC layer maps that
code straight onto the wire, so the TUI can react to *what went wrong* without parsing
English prose, and the message stays free to be as human as it likes.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AdapterError",
    "ConfigError",
    "DatabaseError",
    "DispatchError",
    "IllegalTransition",
    "JobNotFound",
    "MigrationError",
    "QueryError",
    "ValidationError",
]


class DispatchError(Exception):
    """Base class for every error Dispatch raises deliberately.

    Args:
        message: Human-readable explanation. Shown verbatim in the TUI.
        detail: Optional structured payload for programmatic consumers.
    """

    code = "ERROR"

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def __str__(self) -> str:
        return self.message


class ConfigError(DispatchError):
    """The configuration file is malformed or contains an unusable value."""

    code = "CONFIG_INVALID"


class DatabaseError(DispatchError):
    """The database could not be opened, or is missing a required capability."""

    code = "DB_ERROR"


class MigrationError(DatabaseError):
    """A schema migration could not be applied, or the schema is from the future."""

    code = "DB_MIGRATION"


class JobNotFound(DispatchError):
    """No job exists with the given identifier."""

    code = "JOB_NOT_FOUND"

    def __init__(self, job_id: object) -> None:
        super().__init__(f"No job with id {job_id}", detail={"job_id": str(job_id)})


class IllegalTransition(DispatchError):
    """A job state change was attempted that the state machine forbids.

    Raised rather than silently ignored: a transition the code did not expect means the
    caller's model of the world is wrong, and continuing would corrupt history.
    """

    code = "ILLEGAL_TRANSITION"

    def __init__(self, job_id: object, source: object, target: object) -> None:
        super().__init__(
            f"Cannot move job {job_id} from {source} to {target}",
            detail={"job_id": str(job_id), "from": str(source), "to": str(target)},
        )
        self.source = source
        self.target = target


class ValidationError(DispatchError):
    """A value failed validation before it could be stored."""

    code = "VALIDATION_FAILED"


class QueryError(DispatchError):
    """A search query string could not be parsed."""

    code = "QUERY_INVALID"


class AdapterError(DispatchError):
    """A solver adapter is unusable, unknown, or failed while inspecting a case."""

    code = "ADAPTER_ERROR"
