# Dispatch

A lightweight job scheduler for **one** Linux workstation dedicated to scientific
simulations — primarily CFD.

Dispatch is not Slurm. It manages a single machine, has no users, no permissions, and no
network surface. It exists so that a headless workstation accessed over SSH feels like a
small HPC cluster: submit a case, walk away, and find a complete record of what ran.

```
  dispatch          Textual TUI + CLI, disposable
       │            AF_UNIX socket, newline-delimited JSON
  dispatchd         daemon: scheduler, executor, database
       │
  simulations       setsid'd process groups, outliving both
```

## Status

Under construction, built in phases (see `docs/ARCHITECTURE.md` §14).

| Phase | Component | State |
|---|---|---|
| 1 | Architecture | Complete |
| 2a | `core` + `db` | In progress |
| 2b | Daemon, scheduler, IPC | Not started |
| 3 | TUI | Not started |
| 4–6 | OpenFOAM, SU2, Basilisk adapters | Not started |

## Design goals

* **Nothing interrupts a simulation.** Not a dropped SSH session, not a TUI crash, not a
  daemon restart, not an upgrade.
* **Stay out of the way.** The daemon targets under 35 MB resident and 0.0% idle CPU, so
  the RAM belongs to the solvers.
* **Remember everything.** Every run keeps its logs, its case metadata, its resource
  usage, and a reproducibility record — the git commit, the solver build, the exact
  command — for as long as the database exists.
* **Know nothing about solvers.** The scheduler cannot name OpenFOAM. Solver behaviour
  lives entirely in adapters, which return declarative execution plans rather than
  running anything themselves.

## Requirements

Python 3.13, Linux, and a SQLite built with FTS5 and JSON1 (the norm). `uv` for
development.

## Development

```sh
uv venv --python 3.13
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
uv run mypy
```

## Documentation

`docs/ARCHITECTURE.md` is the design of record: every module, the daemon/TUI protocol, the
adapter interface, the database schema, and a decision log giving the rationale and the
rejected alternative for each significant choice.

## Licence

MIT
