"""Architectural invariants, enforced as tests rather than as discipline.

Layering and dependency rules decay the moment they exist only in a document. Each rule
here corresponds to a claim in ``docs/ARCHITECTURE.md``, and breaking one fails the build
with a message naming the offending import.

These pass trivially today, when half the packages are empty. That is the point: they are
written now so that Phase 2b cannot quietly violate them later.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "dispatch"

# `version` is a leaf module with no dependencies of its own, so every layer may read it.
UNIVERSAL = {"version"}

# Which dispatch subpackages each layer may import. See ARCHITECTURE.md §3.
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "db": {"core"},
    "ipc": {"core"},
    "adapters": {"core"},
    "daemon": {"core", "db", "ipc", "adapters"},
    "tui": {"core", "ipc"},
}


def dispatch_imports(path: Path) -> set[str]:
    """Return the dispatch subpackages a module imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found |= _subpackage(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found |= _subpackage(node.module)
    return found


def _subpackage(module: str) -> set[str]:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "dispatch":
        return {parts[1]}
    return set()


def modules_in(layer: str) -> list[Path]:
    return sorted((SRC / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_only_imports_what_it_is_allowed_to(layer: str) -> None:
    violations: list[str] = []
    for module in modules_in(layer):
        for imported in dispatch_imports(module) - {layer} - ALLOWED[layer] - UNIVERSAL:
            violations.append(f"{module.relative_to(SRC)} imports dispatch.{imported}")
    assert not violations, "Layering violations:\n  " + "\n  ".join(violations)


def test_core_imports_nothing_from_dispatch_but_itself() -> None:
    """The domain layer is the foundation; anything it depended on would be beneath it."""
    for module in modules_in("core"):
        assert dispatch_imports(module) <= {"core"} | UNIVERSAL, (
            f"{module.name} reaches outside core"
        )


def test_the_tui_cannot_reach_the_database_or_the_scheduler() -> None:
    """The TUI is a view over the IPC protocol.

    If it could import `db` or `daemon`, the separation that keeps simulations running
    when the TUI crashes would be one convenient shortcut away from being fiction.
    """
    for module in modules_in("tui"):
        reached = dispatch_imports(module)
        assert "db" not in reached, f"{module.name} imports the database layer"
        assert "daemon" not in reached, f"{module.name} imports the daemon"
        assert "adapters" not in reached, f"{module.name} imports solver adapters"


def test_the_daemon_does_not_import_rich_or_textual() -> None:
    """Presentation libraries cost tens of megabytes resident, in a process that runs for
    months. Keeping them out of the daemon is most of how the <35 MB target is met."""
    offenders: list[str] = []
    for layer in ("daemon", "core", "db", "ipc", "adapters"):
        for module in modules_in(layer):
            source = module.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(module))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in ("rich", "textual"):
                        offenders.append(f"{module.relative_to(SRC)} imports {name}")
    assert not offenders, "Presentation libraries in the daemon's import graph:\n  " + "\n  ".join(
        offenders
    )


def test_the_scheduler_knows_nothing_about_solvers() -> None:
    """The central claim of the architecture, checked the blunt way.

    Solver-specific behaviour belongs entirely in adapters. If the word `foam` ever
    appears in the daemon, some solver knowledge has leaked across the boundary and the
    plugin design has started to rot.
    """
    forbidden = ("foam", "su2", "basilisk", "calculix", "openfoam", "mpirun")
    offenders: list[str] = []
    for module in modules_in("daemon"):
        lowered = module.read_text(encoding="utf-8").lower()
        for word in forbidden:
            if word in lowered:
                offenders.append(f"{module.relative_to(SRC)} mentions {word!r}")
    assert not offenders, "Solver knowledge has leaked into the scheduler:\n  " + "\n  ".join(
        offenders
    )


SQL_STATEMENT = re.compile(
    r"\bSELECT\b[^\n]+\bFROM\b"
    r"|\bINSERT\s+(?:OR\s+\w+\s+)?INTO\b"
    r"|\bUPDATE\s+\w+\s+SET\b"
    r"|\bDELETE\s+FROM\b"
    r"|\bCREATE\s+(?:VIRTUAL\s+)?(?:TABLE|INDEX)\b"
)
"""Matches real SQL statements, not prose.

Deliberately shaped rather than a bare keyword list: "Dispatch will only update
numberOfSubdomains" is English, and a check that flags it teaches people to ignore the
check.
"""


def test_only_the_repository_contains_sql() -> None:
    """All SQL lives in one place, so a schema change has one file to touch."""
    allowed = {"db/repository.py", "db/connection.py"}
    offenders: list[str] = []
    for layer in ALLOWED:
        for module in modules_in(layer):
            relative = str(module.relative_to(SRC))
            if relative in allowed:
                continue
            for match in SQL_STATEMENT.finditer(module.read_text(encoding="utf-8")):
                offenders.append(f"{relative} contains SQL: {match.group(0)[:50]!r}")
    assert not offenders, "SQL outside the repository:\n  " + "\n  ".join(offenders)


def test_every_module_has_a_docstring() -> None:
    missing = [
        str(module.relative_to(SRC))
        for layer in ALLOWED
        for module in modules_in(layer)
        if module.name != "__init__.py"
        and ast.get_docstring(ast.parse(module.read_text(encoding="utf-8"))) is None
    ]
    assert not missing, f"Modules without a docstring: {missing}"
