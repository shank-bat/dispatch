"""Getting plottable numbers out of a job, for whoever asked to see them.

The chain the architecture describes runs::

    job -> log parser -> available datasets -> X/Y selection -> terminal renderer

This module is the second and third arrows, and the whole reason it exists is to be the
*only* place they meet. It reads a log off the disk and hands the text to the job's
adapter, which is where every piece of solver knowledge lives; it does not know what a
residual is, and neither does anything else in the daemon.

Three properties are load-bearing:

**It runs when a person asks, and never otherwise.** No timer reads logs for plots, no
cache is kept warm, nothing is parsed at job completion on the chance somebody might look
later. Pressing ``p`` is the only thing that causes any of this to happen, which is the
same demand-driven rule the dashboard sampler follows (§6.1).

**It cannot change a job's state.** Nothing here writes to the database, and the caller
gets numbers or an empty result. A parser that misreads a line produces a wrong point on a
chart; it can never turn a completed run into a failed one (§13.25).

**It is bounded.** A log is read up to a configured size from its end, and every series is
downsampled before it goes on the wire, so plotting a month-old run that produced four
gigabytes of residuals costs a bounded read and a bounded message rather than the daemon's
entire address space.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dispatch.adapters.base import CaseContext
from dispatch.adapters.registry import AdapterRegistry
from dispatch.core.config import PlotConfig
from dispatch.core.models import Job
from dispatch.core.series import PlotData, downsample

__all__ = ["extract_series", "read_log"]

log = logging.getLogger(__name__)


def read_log(path: Path, limit: int) -> tuple[str, bool]:
    """Read up to ``limit`` bytes from the end of a file.

    From the end rather than the beginning: if only part of a long run can be read, the
    recent part is the one worth having -- it is where the run diverged, stalled, or
    converged. The caller is told when this happened so it can say so.

    Returns:
        The text, and whether it is only the tail of a larger file.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        log.debug("Cannot stat %s: %s", path, exc)
        return "", False

    truncated = size > limit
    try:
        with path.open("rb") as handle:
            if truncated:
                handle.seek(-limit, os.SEEK_END)
                # The seek lands mid-line, and a half line at the start of the text would
                # be parsed as a whole one. Dropping to the next newline costs one partial
                # record and removes a whole class of wrong first data point.
                handle.readline()
            data = handle.read()
    except OSError as exc:
        log.debug("Cannot read %s: %s", path, exc)
        return "", False

    return data.decode("utf-8", errors="replace"), truncated


def extract_series(
    job: Job, *, registry: AdapterRegistry, config: PlotConfig
) -> PlotData:
    """Read a job's log and ask its adapter what is plottable in it.

    Never raises. Every failure -- a job with no log yet, a deleted file, an adapter that
    is no longer installed, a parser that throws on a corrupt line -- produces empty plot
    data, which the interface reports as "nothing to plot here". Refusing to open a chart
    is a fine outcome; a traceback out of a keypress is not.

    Args:
        job: The job to read.
        registry: Adapters, for the one that ran this job.
        config: Read and point limits.

    Returns:
        The available series, downsampled to the configured point count.
    """
    path = job.output_path
    if path is None:
        return PlotData()

    text, truncated = read_log(path, config.max_log_bytes)
    if not text.strip():
        return PlotData(truncated=truncated)

    try:
        adapter = registry.get(job.solver)
    except Exception as exc:
        # History outlives adapters: a job run by a plugin that has since been uninstalled
        # is still in the database and still openable, it simply has nothing to plot.
        log.debug("No adapter %r to parse job %s: %s", job.solver, job.id[:8], exc)
        return PlotData(truncated=truncated)

    ctx = CaseContext(
        workdir=job.workdir,
        cores=job.cores,
        ram_mb=job.ram_estimate_mb,
        gpus=job.gpus,
        env=dict(os.environ),
        job_name=job.name,
        metadata=job.metadata,
    )
    try:
        data = adapter.parse_series(text, ctx)
    except Exception:
        log.exception("Adapter %s failed to parse series for job %s", job.solver, job.id[:8])
        return PlotData(truncated=truncated)

    return PlotData(
        series=tuple(downsample(item, config.max_points) for item in data.series if len(item)),
        samples=data.samples,
        truncated=truncated or data.truncated,
    )
