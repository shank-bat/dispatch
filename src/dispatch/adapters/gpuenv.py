"""Telling a solver process which accelerators it may use.

Every framework worth scheduling reads the same environment variable, and every one of
them does the same thing when it is unset: takes whatever it finds. That default is wrong
in both directions on a shared workstation. A CPU job that happens to import a GPU-capable
library will quietly initialise a context on a device another job is using, and a GPU job
gets no signal at all about how many devices it was granted.

So the rule is stated explicitly for both kinds of job, and it is stated **here**, in one
module, rather than in each adapter. That is the same containment the rest of Dispatch
applies to vendor knowledge: :func:`~dispatch.core.config.installed_gpus` is the only
place that knows how to *count* devices, and this is the only place that knows how to
*hide* them. Supporting another runtime means adding a variable to
:data:`GPU_VISIBILITY_VARS`, not touching an adapter.

What this deliberately does **not** do is choose device indices. See §13.24: the ledger
grants a GPU *count*, so a job that was granted one GPU is told it may use a GPU, not
which one. Pinning indices needs the assignment to survive a daemon restart -- otherwise a
re-adopted job and a newly admitted one can be handed the same device -- and that is a
column, a migration, and a recovery path for a feature nobody has asked for yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from dispatch.adapters.base import CaseContext

__all__ = ["GPU_VISIBILITY_VARS", "apply_gpu_visibility", "gpu_environment"]

GPU_VISIBILITY_VARS: Final[tuple[str, ...]] = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)
"""Variables that mean "these are the devices you may use", across runtimes.

CUDA's name is honoured by PyTorch, TensorFlow, JAX, and CuPy alike; the two ROCm
spellings cover AMD. All of them read an empty string as "no devices", which is exactly
the statement a CPU job needs to make.
"""

NO_DEVICES: Final = ""
"""What every listed runtime reads as "you have been given no accelerator"."""


def gpu_environment(ctx: CaseContext) -> Mapping[str, str]:
    """The accelerator-visibility variables a job's commands should carry.

    Two cases, and the CPU one is the one that earns this module its place:

    * **No GPUs granted.** Every variable is set to the empty string, so a CPU job cannot
      wander onto a device that a GPU job is using. This is what makes "a CPU job does not
      accidentally reserve GPUs" true of the *process*, not merely of the ledger.
    * **GPUs granted.** Whatever the daemon's environment already said is passed through
      untouched -- including a machine-wide restriction an administrator set. Overwriting
      it with a guess at device indices would be worse than leaving it alone (§13.24).

    Returns:
        Variables to merge into the step's environment. Possibly empty.
    """
    if ctx.uses_gpu:
        return {
            name: ctx.env[name] for name in GPU_VISIBILITY_VARS if ctx.env.get(name) is not None
        }
    return dict.fromkeys(GPU_VISIBILITY_VARS, NO_DEVICES)


def apply_gpu_visibility(env: dict[str, str], ctx: CaseContext) -> dict[str, str]:
    """Merge :func:`gpu_environment` into ``env`` in place, and return it.

    The shape adapters actually want, since they are already building a mutable copy of
    ``ctx.env`` for their step.
    """
    env.update(gpu_environment(ctx))
    return env


def describe(ctx: CaseContext) -> str:
    """One line explaining what the job will be allowed to use, for a validation note."""
    if not ctx.uses_gpu:
        return (
            "no GPU was requested, so this job runs with "
            f"{GPU_VISIBILITY_VARS[0]} empty and cannot use one"
        )
    return f"{ctx.gpus} GPU(s) reserved; the job's frameworks will see the machine's devices"


def visibility_of(env: Mapping[str, str]) -> Sequence[str]:
    """Which visibility variables an environment sets. For tests and diagnostics."""
    return tuple(name for name in GPU_VISIBILITY_VARS if name in env)
