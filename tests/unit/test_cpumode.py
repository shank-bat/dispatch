"""Physical cores versus logical threads.

The claim under test is that the choice is honoured consistently: the ledger, admission,
the snapshot the interface renders, and the execution plan a launcher receives all agree
about what a requested count means. Disagreement here does not produce a wrong label --
it produces jobs that sit in the queue forever, or ranks a launcher refuses.

Nothing here reads this machine's real topology except the two tests that say so: the
counts are injected, so the suite gives the same answer on a 4-core laptop and a 96-core
node.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.adapters import mpi
from dispatch.adapters.base import CaseContext
from dispatch.core.config import (
    CpuMode,
    SchedulerConfig,
    logical_cpus,
    physical_cores,
)
from dispatch.core.errors import ConfigError
from dispatch.core.models import ResourceRequest
from dispatch.core.validation import ReportBuilder, Severity
from dispatch.daemon.resources import ResourceModel

# -- the mode itself ------------------------------------------------------------------


def test_physical_is_the_default() -> None:
    """An existing installation must schedule identically after this setting appeared."""
    assert SchedulerConfig().resolved_cpu_mode is CpuMode.PHYSICAL


def test_the_mode_is_spelled_the_same_in_config_and_code() -> None:
    assert SchedulerConfig(cpu_mode="logical").resolved_cpu_mode is CpuMode.LOGICAL
    assert SchedulerConfig(cpu_mode="PHYSICAL").resolved_cpu_mode is CpuMode.PHYSICAL
    assert SchedulerConfig(cpu_mode=" logical ").resolved_cpu_mode is CpuMode.LOGICAL


def test_an_unknown_mode_is_a_startup_error_listing_the_valid_ones() -> None:
    """Never a silent fallback: a machine counting the wrong thing is found months later."""
    with pytest.raises(ConfigError) as excinfo:
        SchedulerConfig(cpu_mode="hyperthreads")
    assert "physical" in str(excinfo.value) and "logical" in str(excinfo.value)


def test_an_explicit_core_count_wins_over_either_mode() -> None:
    """A user who names a number means it; the mode then only says what it is."""
    for mode in ("physical", "logical"):
        assert SchedulerConfig(total_cores=7, cpu_mode=mode).resolve_total_cores() == 7


def test_the_two_modes_count_different_things_on_an_smt_machine() -> None:
    """Reads the real topology, and only asserts the relationship between the two."""
    physical, logical = physical_cores(), logical_cpus()
    assert logical >= 1
    if physical is None or physical == logical:
        pytest.skip("this machine reports no SMT, so the two counts coincide")
    assert SchedulerConfig(cpu_mode="logical").resolve_total_cores() == logical
    assert SchedulerConfig().resolve_total_cores() == min(physical, logical)


def test_logical_cpus_counts_what_is_available_rather_than_what_exists() -> None:
    """Affinity masks are sparse, so this must count CPUs and never index them."""
    import os

    assert logical_cpus() == len(os.sched_getaffinity(0))
    assert logical_cpus() >= 1


def test_the_physical_count_is_capped_by_what_the_process_may_use(monkeypatch) -> None:
    """A 16-core machine confined to four CPUs by a cgroup has four, not sixteen."""
    monkeypatch.setattr("dispatch.core.config.physical_cores", lambda: 16)
    monkeypatch.setattr("dispatch.core.config.logical_cpus", lambda: 4)
    assert SchedulerConfig().resolve_total_cores() == 4


def test_a_count_describes_itself_in_the_right_unit() -> None:
    assert SchedulerConfig().describe_cores(20) == "20 cores"
    assert SchedulerConfig(cpu_mode="logical").describe_cores(20) == "20 threads"
    assert SchedulerConfig(cpu_mode="logical").describe_cores(1) == "1 thread"


# -- the ledger agrees ------------------------------------------------------------------


def ledger(*, mode: str = "physical", total: int = 8) -> ResourceModel:
    return ResourceModel(
        SchedulerConfig(total_cores=total, reserved_cores=0, cpu_mode=mode, total_gpus=0),
        memory_probe=lambda: 64_000,
    )


def test_the_ledger_carries_the_mode_so_consumers_cannot_disagree() -> None:
    assert ledger().cpu_mode is CpuMode.PHYSICAL
    assert ledger(mode="logical").cpu_mode is CpuMode.LOGICAL


def test_admission_arithmetic_is_unchanged_by_the_mode() -> None:
    """The mode changes what the total *means*, never how the ledger adds up."""
    for mode in ("physical", "logical"):
        model = ledger(mode=mode, total=8)
        model.acquire("a", ResourceRequest(cores=5))
        assert model.free_cores == 3
        assert model.can_admit(ResourceRequest(cores=3))[0]
        assert not model.can_admit(ResourceRequest(cores=4))[0]


def test_the_ledger_labels_its_own_counts() -> None:
    assert ledger().describe_cores(4) == "4 cores"
    assert ledger(mode="logical").describe_cores(4) == "4 threads"


def test_logical_mode_admits_jobs_physical_mode_would_refuse(monkeypatch) -> None:
    """The whole observable point of the setting."""
    monkeypatch.setattr("dispatch.core.config.physical_cores", lambda: 8)
    monkeypatch.setattr("dispatch.core.config.logical_cpus", lambda: 16)

    strict = ResourceModel(SchedulerConfig(reserved_cores=0), memory_probe=lambda: 64_000)
    loose = ResourceModel(
        SchedulerConfig(reserved_cores=0, cpu_mode="logical"), memory_probe=lambda: 64_000
    )
    request = ResourceRequest(cores=12)
    assert not strict.can_admit(request)[0]
    assert loose.can_admit(request)[0]


# -- the snapshot the interface renders --------------------------------------------------


def snapshot(mode: str):
    from dispatch.core.models import SystemSnapshot

    return SystemSnapshot(
        timestamp=0.0,
        hostname="test",
        total_cores=8,
        allocated_cores=0,
        reserved_cores=0,
        cpu_percent=0.0,
        per_core_percent=(),
        total_ram_mb=1000,
        used_ram_mb=100,
        available_ram_mb=900,
        load_average=(0.0, 0.0, 0.0),
        uptime_s=0.0,
        cpu_mode=mode,
    )


def test_the_snapshot_names_its_own_unit() -> None:
    assert snapshot("physical").core_unit == "core"
    assert snapshot("logical").core_unit == "thread"


def test_the_unit_reaches_the_wire() -> None:
    from dispatch.ipc.protocol import encode_snapshot

    assert encode_snapshot(snapshot("logical"))["core_unit"] == "thread"
    assert encode_snapshot(snapshot("logical"))["cpu_mode"] == "logical"


def test_the_meters_label_threads_as_threads() -> None:
    """A machine scheduling threads that said "cores" would mislabel the one number
    every admission decision is made against."""
    from rich.text import Text

    from dispatch.tui.widgets.meters import ResourceMeters

    base = {
        "total_cores": 8, "allocated_cores": 2, "free_cores": 6, "cpu_percent": 10.0,
        "used_ram_mb": 1000, "total_ram_mb": 8000, "load_average": [0.5],
    }
    meters = ResourceMeters()

    meters.snapshot = {**base, "cpu_mode": "physical"}
    rendered = meters.render()
    assert isinstance(rendered, Text) and "cores" in rendered.plain

    meters.snapshot = {**base, "cpu_mode": "logical"}
    rendered = meters.render()
    assert isinstance(rendered, Text) and "threads" in rendered.plain


# -- the execution plan agrees ------------------------------------------------------------


def context(tmp_path: Path, *, cores: int, mode: str) -> CaseContext:
    return CaseContext(workdir=tmp_path, cores=cores, cpu_mode=mode, env={"PATH": "/nonexistent"})


def test_the_context_knows_which_unit_it_was_given(tmp_path: Path) -> None:
    assert not context(tmp_path, cores=4, mode="physical").counts_threads
    assert context(tmp_path, cores=4, mode="logical").counts_threads


def test_oversubscription_is_a_warning_when_cores_were_meant(tmp_path: Path) -> None:
    builder = ReportBuilder()
    mpi.check_slots(context(tmp_path, cores=24, mode="physical"), builder, slots=16)
    report = builder.build()
    assert any(f.code == "mpi_oversubscribed" for f in report.of(Severity.WARNING))


def test_oversubscription_is_merely_noted_when_threads_were_meant(tmp_path: Path) -> None:
    """Otherwise a correctly configured SMT machine warns on every single job."""
    builder = ReportBuilder()
    mpi.check_slots(context(tmp_path, cores=24, mode="logical"), builder, slots=16)
    report = builder.build()
    assert not report.of(Severity.WARNING)
    assert report.findings and "by design" in report.findings[0].message


def test_a_fitting_rank_count_says_nothing_in_either_mode(tmp_path: Path) -> None:
    for mode in ("physical", "logical"):
        builder = ReportBuilder()
        mpi.check_slots(context(tmp_path, cores=8, mode=mode), builder, slots=16)
        assert not builder.build().findings


def test_the_launcher_still_oversubscribes_when_it_must(tmp_path: Path) -> None:
    """The flag is about the launcher's slot count, which is physical whatever Dispatch
    counts -- so 24 ranks on 16 slots needs it in both modes."""
    assert mpi.OVERSUBSCRIBE_FLAG in mpi.launch_argv(24, "icoFoam", "-parallel", slots=16)
    assert mpi.OVERSUBSCRIBE_FLAG not in mpi.launch_argv(8, "icoFoam", "-parallel", slots=16)
