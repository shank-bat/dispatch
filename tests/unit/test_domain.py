"""The pure domain types: tags, metadata, validation, plans, and models.

No database, no clock beyond the injected fake, no filesystem beyond ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.core.errors import ValidationError
from dispatch.core.metadata import (
    CaseMetadata,
    FieldType,
    MetadataField,
    MetadataSpec,
    SpecRef,
)
from dispatch.core.models import Detection, Job, JobSpec, Page, ResourceRequest, SystemSnapshot
from dispatch.core.plan import (
    CommandStep,
    ExecutionPlan,
    FailureAction,
    StepKind,
    StepOutcome,
)
from dispatch.core.provenance import GitInfo, Provenance, hash_environment
from dispatch.core.states import JobState
from dispatch.core.tags import normalise_tag, normalise_tags
from dispatch.core.validation import ReportBuilder, Severity, ValidationReport

# -- tags ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("paper", "paper"),
        ("  PAPER  ", "paper"),
        ("NACA 0018", "naca-0018"),
        ("Re_100000", "re_100000"),
        ("v2.1", "v2.1"),
        ("a  b   c", "a-b-c"),
    ],
)
def test_tag_normalisation(raw: str, expected: str) -> None:
    assert normalise_tag(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "-leading", ".dot", "has:colon", "emoji🎈"])
def test_invalid_tags_are_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        normalise_tag(raw)


def test_a_tag_cannot_contain_the_search_separator() -> None:
    """A tag with a colon would be unsearchable by construction."""
    with pytest.raises(ValidationError):
        normalise_tag("foo:bar")


def test_overlong_tag_is_rejected() -> None:
    with pytest.raises(ValidationError, match="maximum"):
        normalise_tag("x" * 65)


def test_normalise_tags_deduplicates_after_normalising() -> None:
    assert normalise_tags(["Paper", "paper", "PAPER "]) == {"paper"}


def test_normalise_tags_skips_blanks() -> None:
    """A trailing comma in `a, b,` should not be an error the user has to go fix."""
    assert normalise_tags(["a", "", "  ", "b"]) == {"a", "b"}


# -- metadata ----------------------------------------------------------------------------


@pytest.fixture
def spec() -> MetadataSpec:
    return MetadataSpec(
        ref=SpecRef("demo", 1),
        fields=(
            MetadataField("app", FieldType.STR, "Application", display_order=2),
            MetadataField("endTime", FieldType.FLOAT, "End", unit="s", display_order=1),
            MetadataField("cells", FieldType.INT, "Cells"),
            MetadataField("adjoint", FieldType.BOOL, "Adjoint"),
        ),
    )


def test_build_coerces_declared_types(spec: MetadataSpec) -> None:
    meta = spec.build({"endTime": "600", "cells": "2418000", "adjoint": "yes"})
    assert meta.case == {"endTime": 600.0, "cells": 2418000, "adjoint": True}


def test_build_rejects_an_undeclared_key(spec: MetadataSpec) -> None:
    """An adapter emitting an undeclared field is a bug, and failing at write time is
    far cheaper than discovering it when a search silently returns nothing."""
    with pytest.raises(ValidationError, match="undeclared metadata key"):
        spec.build({"mystery": 1})


def test_build_skips_none_rather_than_storing_a_fabricated_value(spec: MetadataSpec) -> None:
    assert "endTime" not in spec.build({"endTime": None, "app": "x"}).case


def test_build_rejects_an_uncoercible_value(spec: MetadataSpec) -> None:
    with pytest.raises(ValidationError, match="expects"):
        spec.build({"cells": "many"})


def test_bool_is_not_accepted_as_a_number(spec: MetadataSpec) -> None:
    """bool is an int subclass, so `cells = True` would otherwise silently store 1."""
    with pytest.raises(ValidationError):
        spec.build({"cells": True})


def test_duplicate_field_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate metadata field"):
        MetadataSpec(
            ref=SpecRef("x", 1),
            fields=(
                MetadataField("a", FieldType.STR, "A"),
                MetadataField("a", FieldType.INT, "A again"),
            ),
        )


def test_ordered_uses_display_order(spec: MetadataSpec) -> None:
    assert [f.key for f in spec.ordered][:2] == ["endTime", "app"]


def test_metadata_round_trips_through_json(spec: MetadataSpec) -> None:
    original = spec.build({"app": "interFoam", "endTime": 600}, raw_dict="anything")
    restored = CaseMetadata.from_json(original.to_json())
    assert restored.case == original.case
    assert restored.extra == {"raw_dict": "anything"}
    assert restored.spec == SpecRef("demo", 1)


def test_from_json_tolerates_garbage() -> None:
    """History written by a past version must stay loadable, not raise."""
    assert CaseMetadata.from_json({}).case == {}
    assert CaseMetadata.from_json({"case": "not a mapping"}).case == {}
    assert CaseMetadata.from_json({"spec": 7}).spec.adapter == ""


def test_searchable_pairs_splits_numeric_and_text(spec: MetadataSpec) -> None:
    meta = spec.build({"app": "interFoam", "endTime": 600, "cells": 100})
    pairs = {key: (text, num) for key, text, num in meta.searchable_pairs(spec)}
    assert pairs["app"] == ("interFoam", None)
    assert pairs["endTime"] == (None, 600.0)
    assert pairs["cells"] == (None, 100.0)


def test_searchable_pairs_infers_types_without_a_spec(spec: MetadataSpec) -> None:
    meta = spec.build({"app": "interFoam", "endTime": 600})
    pairs = {key: (text, num) for key, text, num in meta.searchable_pairs(None)}
    assert pairs["endTime"] == (None, 600.0)


def test_flatten_for_fts_includes_keys_and_values(spec: MetadataSpec) -> None:
    text = spec.build({"app": "interFoam"}, note="extra").flatten_for_fts()
    assert "app" in text and "interFoam" in text and "note" in text and "extra" in text


def test_booleans_flatten_to_lowercase(spec: MetadataSpec) -> None:
    """So a query typed in the obvious lowercase form matches."""
    assert "true" in spec.build({"adjoint": True}).flatten_for_fts()


# -- validation ---------------------------------------------------------------------------


def test_empty_report_passes() -> None:
    report = ValidationReport()
    assert report.passed
    assert report.worst is None
    assert report.summary() == "PASS"


def test_warnings_do_not_block() -> None:
    builder = ReportBuilder()
    builder.warning("decomposeParDict is missing; one will be generated")
    report = builder.build()
    assert report.passed
    assert report.summary() == "PASS (1 warning)"


def test_errors_block() -> None:
    builder = ReportBuilder()
    builder.error("system/controlDict is missing", code="missing_controldict")
    report = builder.build()
    assert not report.passed
    assert report.errors[0].code == "missing_controldict"
    assert report.summary() == "FAIL (1 error)"


def test_summary_counts_each_severity() -> None:
    builder = ReportBuilder()
    builder.error("a")
    builder.error("b")
    builder.warning("c")
    builder.info("d")
    assert builder.build().summary() == "FAIL (2 errors, 1 warning, 1 note)"


def test_worst_is_the_highest_severity() -> None:
    builder = ReportBuilder()
    builder.info("a")
    builder.error("b")
    assert builder.build().worst is Severity.ERROR


def test_reports_merge_in_order() -> None:
    first = ReportBuilder()
    first.info("one")
    second = ReportBuilder()
    second.error("two")
    merged = ValidationReport.merge(first.build(), second.build())
    assert [f.message for f in merged] == ["one", "two"]
    assert not merged.passed


# -- plans ----------------------------------------------------------------------------------


def _solve(argv=("solver",), cwd=Path("/tmp")) -> CommandStep:
    return CommandStep(argv=argv, cwd=cwd, description="run", kind=StepKind.SOLVE)


def test_a_plan_needs_exactly_one_solve_step() -> None:
    """Zero would leave nothing to supervise; two would leave no defensible exit code."""
    with pytest.raises(ValidationError, match="exactly one SOLVE"):
        ExecutionPlan(steps=())
    with pytest.raises(ValidationError, match="exactly one SOLVE"):
        ExecutionPlan(steps=(_solve(), _solve()))


def test_plan_exposes_its_phases_in_order() -> None:
    prep_a = CommandStep(["decomposePar"], Path("/tmp"), "decompose")
    prep_b = CommandStep(["renumberMesh"], Path("/tmp"), "renumber")
    cleanup = CommandStep(["echo"], Path("/tmp"), "note", kind=StepKind.CLEANUP)
    plan = ExecutionPlan(steps=(prep_a, prep_b, _solve(), cleanup))
    assert [s.description for s in plan.prepare] == ["decompose", "renumber"]
    assert plan.solve.kind is StepKind.SOLVE
    assert [s.description for s in plan.cleanup] == ["note"]
    assert len(plan) == 4


def test_empty_argv_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        CommandStep([], Path("/tmp"), "nothing")


def test_non_string_argv_is_rejected() -> None:
    """A stray int in argv would fail deep inside the executor instead of here."""
    with pytest.raises(ValidationError, match="all strings"):
        CommandStep(["mpirun", "-np", 16], Path("/tmp"), "run")  # type: ignore[list-item]


def test_non_positive_timeout_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timeout_s must be positive"):
        CommandStep(["x"], Path("/tmp"), "x", timeout_s=0)


def test_render_quotes_for_display() -> None:
    step = CommandStep(["decomposePar", "-case", "/home/shu/my case"], Path("/tmp"), "d")
    assert step.render() == "decomposePar -case '/home/shu/my case'"


def test_step_outcome_success_and_failure() -> None:
    step = CommandStep(["x"], Path("/tmp"), "x")
    assert StepOutcome(step, exit_code=0, duration_s=1.0).ok
    assert StepOutcome(step, exit_code=1, duration_s=1.0).fatal
    assert StepOutcome(step, exit_code=None, duration_s=1.0, timed_out=True).fatal


def test_a_warn_step_failure_is_not_fatal() -> None:
    step = CommandStep(["x"], Path("/tmp"), "x", on_failure=FailureAction.WARN)
    assert not StepOutcome(step, exit_code=1, duration_s=1.0).fatal


# -- models -----------------------------------------------------------------------------------


def test_resource_request_requires_a_core() -> None:
    with pytest.raises(ValidationError, match="at least one core"):
        ResourceRequest(cores=0)


def test_resource_request_rejects_a_non_positive_ram_estimate() -> None:
    with pytest.raises(ValidationError, match="RAM estimate"):
        ResourceRequest(cores=1, ram_mb=0)


def test_job_spec_normalises_tags_and_expands_the_path(tmp_path: Path) -> None:
    spec = JobSpec(
        workdir=tmp_path / "case",
        solver="openfoam",
        resources=ResourceRequest(cores=1),
        tags=frozenset({"Paper", "NACA 0018"}),
    )
    assert spec.tags == {"paper", "naca-0018"}
    assert spec.workdir.is_absolute()


def test_job_spec_requires_a_solver(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must name a solver"):
        JobSpec(workdir=tmp_path, solver="", resources=ResourceRequest(cores=1))


def test_job_spec_rejects_an_overlong_name(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="maximum"):
        JobSpec(
            workdir=tmp_path,
            solver="x",
            resources=ResourceRequest(cores=1),
            name="n" * 500,
        )


def test_jobs_are_immutable() -> None:
    job = _job()
    with pytest.raises(AttributeError):
        job.state = JobState.RUNNING  # type: ignore[misc]


def test_with_state_returns_a_copy() -> None:
    job = _job()
    moved = job.with_state(JobState.HELD)
    assert moved.state is JobState.HELD
    assert job.state is JobState.QUEUED


def test_elapsed_is_none_before_a_job_starts() -> None:
    assert _job().elapsed(now=1000.0) is None


def test_elapsed_advances_while_running_and_freezes_when_finished() -> None:
    running = _job(state=JobState.RUNNING, started_at=100.0)
    assert running.elapsed(now=160.0) == 60.0
    finished = _job(state=JobState.COMPLETED, started_at=100.0, finished_at=150.0)
    assert finished.elapsed(now=99_999.0) == 50.0


def test_detection_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        Detection(solver="openfoam", confidence=1.5)


def test_system_snapshot_free_cores_accounts_for_the_reservation() -> None:
    snapshot = _snapshot(total_cores=24, allocated_cores=16, reserved_cores=1)
    assert snapshot.free_cores == 7


def test_free_cores_never_goes_negative() -> None:
    """Over-allocation is possible after a config change that lowers the core count."""
    assert _snapshot(total_cores=4, allocated_cores=8, reserved_cores=1).free_cores == 0


def test_ram_percent() -> None:
    assert _snapshot(total_ram_mb=1000, used_ram_mb=250).ram_percent == 25.0


def test_page_reports_whether_more_remain() -> None:
    assert Page(items=(1, 2), total=5, offset=0).has_more
    assert not Page(items=(4, 5), total=5, offset=3).has_more


# -- provenance ---------------------------------------------------------------------------------


def test_environment_hash_is_order_independent() -> None:
    assert hash_environment({"A": "1", "B": "2"}) == hash_environment({"B": "2", "A": "1"})


def test_environment_hash_distinguishes_a_shifted_delimiter() -> None:
    """`{"A": "1=B"}` and `{"A=1": "B"}` must not collide."""
    assert hash_environment({"A": "1=B"}) != hash_environment({"A=1": "B"})


def test_git_describe_reports_dirtiness() -> None:
    info = GitInfo(commit="a1b2c3d4e5", branch="main", dirty=True)
    assert info.describe() == "a1b2c3d (main, dirty)"


def test_git_describe_without_a_repository() -> None:
    assert GitInfo().describe() == "not a git repository"
    assert not GitInfo().is_present


def test_provenance_serialises_to_json() -> None:
    prov = Provenance(
        captured_at=1.0,
        dispatch_version="0.1.0",
        python_version="3.13.0",
        hostname="eddy",
        kernel_version="6.16.3",
        git=GitInfo(commit="abc", dirty=True),
        argv=("mpirun", "-np", "16", "interFoam"),
    )
    payload = prov.to_json()
    assert payload["git"]["dirty"] is True
    assert payload["argv"] == ["mpirun", "-np", "16", "interFoam"]


# -- helpers ---------------------------------------------------------------------------------------


def _job(**overrides) -> Job:
    defaults = {
        "id": "00000000-0000-0000-0000-000000000000",
        "seq": 1,
        "name": "case",
        "workdir": Path("/tmp/case"),
        "solver": "openfoam",
        "resources": ResourceRequest(cores=4),
        "state": JobState.QUEUED,
        "created_at": 0.0,
    }
    return Job(**{**defaults, **overrides})


def _snapshot(**overrides) -> SystemSnapshot:
    defaults = {
        "timestamp": 0.0,
        "hostname": "eddy",
        "total_cores": 24,
        "allocated_cores": 0,
        "reserved_cores": 1,
        "cpu_percent": 0.0,
        "per_core_percent": (),
        "total_ram_mb": 32000,
        "used_ram_mb": 8000,
        "available_ram_mb": 24000,
        "load_average": (0.0, 0.0, 0.0),
        "uptime_s": 0.0,
    }
    return SystemSnapshot(**{**defaults, **overrides})
