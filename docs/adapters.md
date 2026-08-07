# Writing a solver adapter

A solver adapter answers four questions about a directory — *is this mine*, *is it valid*,
*what should I record about it*, and *what commands would run it* — and never executes
anything itself.

That last constraint is the whole design. Adapters return an `ExecutionPlan`, which is
data. The daemon runs it, logs it, times it out, supervises it, and cancels it — once, for
every solver that will ever exist. Two things fall out of that for free:

* **Your adapter gets dry run for free**, and it cannot drift, because `--dry-run` is the
  real submission path minus the final call.
* **Your adapter is testable with a temporary directory and no solver installed.** Assert
  on the `argv` lists it produces. The built-in adapters' tests never invoke OpenFOAM.

## The minimum

Three methods. Everything else has a defensible default on `BaseAdapter`.

```python
from pathlib import Path
from typing import ClassVar

from dispatch.adapters.base import BaseAdapter, CaseContext
from dispatch.core.models import Detection
from dispatch.core.plan import CommandStep, ExecutionPlan, StepKind
from dispatch.core.validation import ReportBuilder, ValidationReport


class MySolverAdapter(BaseAdapter):
    """Runs MySolver cases."""

    name: ClassVar[str] = "mysolver"
    display_name: ClassVar[str] = "MySolver"

    @classmethod
    def detect(cls, path: Path) -> Detection | None:
        """Cheap and read-only: this runs on every directory the user browses to."""
        if not (path / "mysolver.in").is_file():
            return None
        return Detection(
            solver=cls.name,
            confidence=0.9,
            solver_binary="mysolver",
            label="MySolver case",
            entry=path / "mysolver.in",
        )

    def validate(self, ctx: CaseContext) -> ValidationReport:
        builder = ReportBuilder()
        if ctx.which("mysolver") is None:
            builder.error("mysolver is not on the PATH", code="solver_not_found")
        return builder.build()

    def plan(self, ctx: CaseContext) -> ExecutionPlan:
        return ExecutionPlan(
            steps=(
                CommandStep(
                    argv=["mysolver", "mysolver.in"],
                    cwd=ctx.workdir,
                    description="Running MySolver",
                    kind=StepKind.SOLVE,
                    env=dict(ctx.env),
                ),
            )
        )
```

Register it in `dispatch/adapters/registry.py`, or ship it as a package:

```toml
[project.entry-points."dispatch.adapters"]
mysolver = "dispatch_mysolver:MySolverAdapter"
```

## Confidence

`Detection.confidence` decides what happens when two adapters claim the same directory.
Be honest about the strength of your evidence — that is what the number is for.

| Evidence | Confidence | Why |
|---|---|---|
| A file that means one thing (`system/controlDict`) | 0.95 | Unambiguous |
| A config whose *contents* were checked (`*.cfg` with `SOLVER=`) | 0.85 | Extension is weak; content is not |
| A deck keyword (`*.inp` containing `*STEP`) | 0.80 | Shared extension, distinctive syntax |
| A source file including known headers (`*.c` with Basilisk headers) | 0.70 | `.c` files are everywhere |

A clear winner is used silently — the user is never asked which solver to use when
detection succeeds. Only a tie prompts.

## Preparation steps

Anything that has to happen before the solver runs is a `PREPARE` step. Mesh decomposition,
source compilation, field initialisation — the executor cannot tell them apart, runs them
in order, records their output in the job's step transcript, and fails the job if one
aborts.

Compilation is the interesting case, because it shows the boundary is in the right place.
The Basilisk adapter compiles with `qcc` before running, and supporting it required no
scheduler change at all:

```python
def plan(self, ctx: CaseContext) -> ExecutionPlan:
    return ExecutionPlan(steps=(
        CommandStep(argv=["qcc", "-O2", "-o", "sim", "sim.c", "-lm"],
                    cwd=ctx.workdir, description="Compiling with qcc",
                    kind=StepKind.PREPARE, timeout_s=600),
        CommandStep(argv=["./sim"], cwd=ctx.workdir,
                    description="Running sim", kind=StepKind.SOLVE),
    ))
```

Use `on_failure=FailureAction.WARN` for a step that may legitimately fail. The OpenFOAM
adapter does this for `reconstructPar`, because a case that was decomposed but never run
has nothing to reconstruct and losing the job over that would be wrong.

A plan needs **exactly one** `SOLVE` step. Zero would leave nothing to supervise; two would
leave the executor with no defensible answer to "which exit code is the job's".

## Parallelism is yours to define

Nothing in the interface assumes MPI:

* OpenFOAM decomposes the mesh and launches `mpirun -np N`.

* SU2 partitions internally, so only the launcher changes.
* Basilisk needs `-D_MPI=N` at *compile* time, matching the rank count.
* CalculiX uses OpenMP, so `ctx.cores` becomes `OMP_NUM_THREADS` in the step's `env`.

The three that do use MPI share `adapters/mpi.py` rather than each writing the `mpirun`
line themselves:

```python
argv = mpi.launch_argv(ctx.cores, application, "-parallel")
```

`ctx.cores` is a count of **physical** cores, because that is what MPI's default slot count
means. Building the argv by hand and asking for more ranks than the machine has slots does
not run slowly — the launcher refuses outright and the solver never starts.

## Declaring metadata

A bare dictionary is where information goes to become unsearchable. Declare what you
extract and it becomes typed, searchable, and correctly labelled in the interface:

```python
metadata_spec: ClassVar[MetadataSpec] = MetadataSpec(
    ref=SpecRef(adapter="mysolver", version=1),
    fields=(
        MetadataField("iterations", FieldType.INT, "Iterations", display_order=1),
        MetadataField("endTime", FieldType.FLOAT, "End time", unit="s", display_order=2),
    ),
)

def collect_metadata(self, ctx: CaseContext) -> CaseMetadata:
    return self.metadata_spec.build({"iterations": 500, "endTime": 600.0})
```

The user can then search `endTime>500` and get a numeric comparison rather than a string
one. Emitting a key you did not declare is an error at write time — far cheaper than
discovering it when a search silently returns nothing.

Bump `SpecRef.version` if you rename or retype a field; stored history keeps the version it
was written against.

## Environments that need sourcing

If your solver only exists after a setup script has been sourced, use `shellenv.capture`.
It sources the script **once per daemon lifetime** and reuses the result:

```python
def prepare_environment(self, ctx: CaseContext) -> Mapping[str, str]:
    script = self.setting("setup_script")
    return capture(Path(script), base=dict(ctx.env)) if script else ctx.env
```

Use `ctx.which(...)` rather than `shutil.which(...)` in validation, so the check runs
against the PATH the job will actually have.

## Optional methods worth implementing

| Method | Why |
|---|---|
| `parse_progress(tail, ctx)` | Extract the current step from the log's last 8 KB. Returning `None` is fine — the interface shows elapsed time instead. |
| `stop_gracefully(ctx)` | A solver-native clean stop. OpenFOAM writes `stopAt writeNow;` so a cancelled run leaves a usable result rather than a truncated one. Return `False` for the signal ladder. |
| `finalize(ctx)` | Undo case edits that steered *this* run. Pair it with `stop_gracefully` — see below. |
| `explain_failure(tail, ctx)` | Turn the end of a failed job's output into the sentence a user actually wants. The default reads the last meaningful lines and is decent; override it if your solver has a recognisable error format. |
| `solver_version(ctx)` | Recorded in every job's provenance. Cached per daemon lifetime. |
| `suggest_tags(ctx)` | Offered at submission, never applied silently. |
| `env_keys` | Environment variables worth recording verbatim in provenance. Everything else is reduced to a hash. |

## Rules

* **Never spawn a process.** `plan()` describes; it does not act. Breaking this loses dry
  run, step transcripts, timeouts, and cancellation for your adapter.
* **`detect()` must be cheap and read-only.** It runs for every adapter on every directory
  the user browses to.
* **Do not raise from `validate()`.** Return findings. A crash is caught and turned into an
  ERROR finding, but a real finding is more useful than a traceback.
* **`plan()` may write configuration files.** The OpenFOAM adapter rewrites
  `decomposeParDict` here. Say so in a validation finding first — silently changing a
  user's case is not acceptable, telling them and then doing it is.
* **If you edit a case to control a run, undo it in `finalize()`.** This is the rule with
  the nastiest failure mode in the whole interface. OpenFOAM's clean stop sets
  `stopAt writeNow` in `controlDict`; leave it there and every *later* run of that case
  writes once and exits at its first time step — with status 0, so the job is recorded as
  completed, no error appears anywhere, and the case simply seems to have stopped working.
  `finalize()` runs however the job ended, including cancellation, so it is the right place.
  Keep it to a couple of file operations: it is called synchronously, from a `finally`, on a
  task that may already be cancelling.
* **`finalize()` is not for cleaning up results.** A cancelled run's output belongs to the
  user. Deleting any of it is not your adapter's decision.
* **Do not build `mpirun` argv by hand.** Use `adapters/mpi.py`. It reconciles the rank
  count with the launcher's slot count, which is not a solver concern and which every
  adapter would otherwise get wrong in the same way (§8.7 of ARCHITECTURE.md).

## Versioning

`api_version` must equal `ADAPTER_API_VERSION`. A mismatch is refused with a message naming
both versions, and **the daemon still starts** with the remaining adapters: one stale
plugin must not take three months of queued work down with it. `dispatch info` lists what
loaded and what did not.

`adapter_version` is your own revision counter. It is recorded in each job's provenance, so
"this run was produced by rev 3 of the adapter" stays answerable.

## Testing

Assert on plans as data:

```python
def test_parallel_plan(tmp_path):
    case = make_case(tmp_path)
    plan = MySolverAdapter({}).plan(CaseContext(workdir=case, cores=8, env={"PATH": "/nonexistent"}))
    assert [list(s.argv) for s in plan.steps] == [["mpirun", "-np", "8", "mysolver", "in"]]
```

Pass a `PATH` that finds nothing so `which()` checks are deterministic on any machine. See
`tests/unit/test_adapters.py` for the full pattern.
