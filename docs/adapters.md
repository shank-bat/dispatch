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
    log_name: ClassVar[str] = "log.mysolver"

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

## Naming the log

`log_name` is the file your solver's output goes into, **inside the case directory**:

```
~/projects/cavity/
├── system/
├── constant/
└── log.foam        <- written by the kernel, straight from the solver
```

Follow the convention your solver's users already have — it has to be recognisable to somebody
browsing the case who has never heard of Dispatch. `log.foam`, `log.su2`, `log.calculix`.

Three things worth knowing:

* **stdout and stderr both go there**, appended to one file, in order. That is what a user
  produces by hand with `> log.foam 2>&1`, and it is what they read.
* **The daemon opens it, not you.** You supply a string; the executor handles rotation, the
  fallback when the case directory is unwritable, and the file descriptors.
* **A second run rotates the first aside** to `log.foam.1`, and the earlier job's history follows
  it. You do not have to make the name unique, and you should not try.

Omit `log_name` and you get `log.job`. Adapters written before this existed keep working.

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
| `parse_series(text, ctx)` | Numerical series a user can plot, read from the job's output. See below. |
| `stop_gracefully(ctx)` | A solver-native clean stop. OpenFOAM writes `stopAt writeNow;` so a cancelled run leaves a usable result rather than a truncated one. Return `False` for the signal ladder. |
| `finalize(ctx)` | Undo case edits that steered *this* run. Pair it with `stop_gracefully` — see below. |
| `explain_failure(tail, ctx)` | Turn the end of a failed job's output into the sentence a user actually wants. The default reads the last meaningful lines and is decent; override it if your solver has a recognisable error format. |
| `solver_version(ctx)` | Recorded in every job's provenance. Cached per daemon lifetime. |
| `suggest_tags(ctx)` | Offered at submission, never applied silently. |
| `env_keys` | Environment variables worth recording verbatim in provenance. Everything else is reduced to a hash. |

## Exposing plottable data

`p` on a job asks the daemon for its numbers, and the daemon asks you. Return the series your
log actually contains:

```python
from dispatch.core.series import PlotData, series_from_records

def parse_series(self, text: str, ctx: CaseContext) -> PlotData:
    records: list[dict[str, float]] = []
    for line in text.splitlines():
        if match := STEP_LINE.match(line):
            records.append({"iteration": float(match["n"]), "residual": float(match["r"])})
    return PlotData(
        series=series_from_records(records, axes=["iteration"]),
        samples=len(records),
    )
```

`series_from_records` takes the shape every log parser naturally produces — one mapping per step,
holding whatever that step printed — and turns it into columns, recording for each value *which
step it came from*. That last part matters: a quantity printed on every step and one printed
sporadically have different lengths, and the interface pairs them by sample index rather than
zipping them, so a gap never plots one quantity against the wrong other one.

Four rules:

* **Emit only what appeared.** A 2-D case has no `Uz`, so it must produce no `residual(Uz)` — not
  an empty one, and not one full of zeros. `PlotData.get` returning `None` is how "this log does
  not have that" is expressed.
* **Mark the axis-ish series.** `axes=[...]` flags quantities that increase monotonically —
  iteration, time, wall-clock. It only orders the selector; the user may put anything on either
  axis, and `execution_time` vs `iteration` is a plot people genuinely want.
* **Never raise.** A log is the least trustworthy text in the system: truncated mid-line by a
  kill, interleaved by `mpirun`, full of `nan` from a diverging run. Each of those must produce a
  shorter series, never an exception. Drop non-finite values — they poison every axis calculation
  downstream, and the divergence stays visible in the points before them.
* **Return `PlotData()` when you have nothing.** It is a first-class answer, and the interface
  says so rather than drawing an empty chart.

You are handed the log's text, or its last `plot.max_log_bytes` when it is very large. This runs
only when a user presses `p`, never on a timer, and **nothing about a job's state depends on what
you return** — a misparsed line costs a wrong point on a chart and can never make a completed run
look failed.

`adapters/foamlog.py` is the reference implementation.

## GPUs

`ctx.gpus` is how many GPUs the scheduler reserved for this job. A count, not device indices.

Use `adapters/gpuenv.py` rather than setting `CUDA_VISIBLE_DEVICES` yourself:

```python
env = gpuenv.apply_gpu_visibility(dict(ctx.env), ctx)
```

For a job granted no GPUs this sets every visibility variable empty, so CPU work cannot wander
onto a device another job is using — which is what makes the ledger's promise true of the process
too. For a job granted GPUs it passes the inherited environment through untouched. Every built-in
adapter calls it, CFD ones included; the rule is machine-wide, not a property of the ML adapters.

That module is the only place in Dispatch that knows how to hide a device, exactly as
`config.installed_gpus()` is the only place that knows how to count one. Supporting another
runtime means adding a variable name there, not editing an adapter.

## Python projects: the `dispatch.toml` convention

If your workload is "run this Python program", the `ml` and `pinn` adapters may already cover it,
and both read an explicit declaration from the project directory:

```toml
# dispatch.toml
[job]
adapter    = "pinn"          # which adapter owns this directory
entrypoint = "train.py"      # or a module, with module = true
args       = ["--config", "configs/burgers.yaml"]
venv       = ".venv"         # its bin/python wins over `python`
python     = "python3.12"    # fallback interpreter
framework  = "deepxde"       # recorded as metadata, suggested as a tag
module     = false
```

Detection sees it at confidence 0.95 and every other adapter stays quiet: the user has already
answered the question. Without it, `ml` needs `train.py` *and* a dependency manifest, and `pinn`
needs an actual PINN library in the project's imports or requirements. Neither claims a directory
on the strength of `pyproject.toml`, `main.py`, or `import torch`, and neither guesses from
directory names.

If you write an adapter for another declarative Python workload, reuse
`pyjob.PythonJobAdapter`: you inherit entrypoint resolution, interpreter discovery, the
one-command plan, GPU visibility, and training-curve parsing, and you write `detect` and a
`log_name`.

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
* **Do not set `CUDA_VISIBLE_DEVICES` by hand.** Use `adapters/gpuenv.py`, for the same reason as
  `mpi.py` below: it is a scheduler concern rather than a solver one, and every adapter would
  otherwise get it wrong in the same way.
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
