"""Logging configuration for the daemon.

Rotation is mandatory, not optional. This process is expected to run for months, and an
unrotated log on the same filesystem as the simulation output is a slow-motion disk-full
failure that would eventually take a job down with it.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

__all__ = ["configure_logging"]

FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 8 * 1024 * 1024
BACKUP_COUNT = 3


def configure_logging(
    *,
    log_file: Path | None = None,
    level: int | str = logging.INFO,
    console: bool = True,
) -> None:
    """Set up the root logger.

    Args:
        log_file: Rotating file to write to. ``None`` logs only to the console.
        level: Threshold, as a level or its name.
        console: Whether to also write to stderr. Left on in foreground mode so that
            ``journalctl`` and ``tmux`` both show something useful; turned off when
            detaching, where stderr is already redirected to the file.
    """
    root = logging.getLogger()
    root.setLevel(_resolve(level))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    # asyncio's debug chatter is not useful here and is voluminous under load.
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def _resolve(level: int | str) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(str(level).upper())
    return resolved if resolved is not None else logging.INFO
