"""Launching MPI solvers so that the launcher agrees with the scheduler.

Every parallel adapter here -- OpenFOAM, SU2, Basilisk -- ultimately runs ``mpirun -np N``.
That single line hides a disagreement that costs a job its entire run:

**Dispatch and Open MPI count cores differently.** Open MPI sizes its default slot count by
*physical* cores. ``os.cpu_count()`` reports *logical* CPUs, which on any SMT machine is
larger -- twice as large on a typical desktop part. A scheduler that admits against the
logical count therefore hands ``mpirun`` more ranks than it has slots, and Open MPI refuses
to launch at all::

    There are not enough slots available in the system to satisfy the 23
    slots that were requested by the application

The solver never starts, so the job fails in under a second with no solver output to
explain it -- it looks like the *case* is broken when nothing is wrong with it.

:func:`~dispatch.core.config.SchedulerConfig.resolve_total_cores` closes the gap from the
scheduling side by admitting against physical cores. This module closes it from the launch
side, for the case where the user has deliberately overridden ``scheduler.total_cores``
upwards: rather than failing at the starting line, the rank count they asked for is
launched with ``--oversubscribe``, which is what "I want 24 ranks on 16 cores" means.

See ``docs/ARCHITECTURE.md`` §8.7.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from dispatch.adapters.base import CaseContext
from dispatch.core.config import physical_cores
from dispatch.core.validation import ReportBuilder

__all__ = ["OVERSUBSCRIBE_FLAG", "check_slots", "launch_argv", "slots_available"]

log = logging.getLogger(__name__)

OVERSUBSCRIBE_FLAG = "--oversubscribe"
"""Understood by Open MPI and MPICH alike; both treat it as "trust my rank count"."""


def slots_available() -> int | None:
    """How many ranks ``mpirun`` will launch without being asked twice.

    Physical cores, which is Open MPI's own default. ``None`` when undeterminable, which
    callers treat as "do not second-guess the launcher".
    """
    return physical_cores()


def launch_argv(
    cores: int,
    program: str,
    *args: str,
    slots: int | None = None,
) -> list[str]:
    """Build the ``mpirun`` command line for ``cores`` ranks.

    Adds ``--oversubscribe`` only when the requested rank count genuinely exceeds the
    machine's slots. It is not added unconditionally: the flag also disables the launcher's
    guard against a *mistaken* rank count, and on a correctly-sized job that guard is worth
    keeping.

    Args:
        cores: Rank count. Must be at least 2 -- a serial run should not go through MPI.
        program: The solver binary.
        args: Solver arguments, e.g. ``-parallel``.
        slots: Override the detected slot count. For tests.

    Returns:
        The full argv, starting with ``mpirun``.
    """
    argv = ["mpirun", "-np", str(cores)]

    available = slots_available() if slots is None else slots
    if available is not None and cores > available:
        # The scheduler admitted this, so the core count is the user's deliberate choice
        # (an explicit scheduler.total_cores). Honour it rather than dying at launch.
        log.info(
            "Requesting %d MPI ranks on a machine with %d slots; adding %s",
            cores,
            available,
            OVERSUBSCRIBE_FLAG,
        )
        argv.append(OVERSUBSCRIBE_FLAG)

    argv.append(program)
    argv.extend(args)
    return argv


def check_slots(ctx: CaseContext, builder: ReportBuilder, *, slots: int | None = None) -> None:
    """Warn when a parallel run will oversubscribe the machine.

    A warning rather than an error: the job *will* run, thanks to
    :func:`launch_argv`. What it will not do is run well, and saying so at submission is
    considerably more use than leaving the user to wonder why 24 ranks are slower than 16.
    """
    if ctx.cores < 2:
        return
    available = slots_available() if slots is None else slots
    if available is None or ctx.cores <= available:
        return
    builder.warning(
        f"{ctx.cores} ranks were requested but this machine has {available} physical "
        f"cores, so the ranks will share cores and the run will be slower than "
        f"{available} would be",
        hint=(
            "Request at most one rank per physical core. Dispatch schedules against "
            "physical cores unless scheduler.total_cores overrides it."
        ),
        code="mpi_oversubscribed",
    )


def mpi_program(argv: Sequence[str]) -> str | None:
    """The solver binary out of an ``mpirun`` command line, skipping its flags.

    Used for display and for PATH checks, where the interesting program is the solver
    rather than the launcher.
    """
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "mpirun":
            index += 1
        elif token in ("-np", "-n", "--np", "-c"):
            index += 2
        elif token.startswith("-"):
            index += 1
        else:
            return token
    return None
