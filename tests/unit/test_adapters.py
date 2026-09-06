"""Solver adapters: detection, validation, and the plans they build.

Plans are asserted **as data** -- lists of ``argv`` -- so none of this needs OpenFOAM, SU2,
Basilisk, or CalculiX installed. That is a direct consequence of adapters returning plans
rather than running commands, and it is why these tests run in milliseconds on any machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.adapters import foamdict, mpi
from dispatch.adapters.base import (
    ADAPTER_API_VERSION,
    BaseAdapter,
    CaseContext,
    generic_failure_summary,
)
from dispatch.adapters.basilisk import BasiliskAdapter
from dispatch.adapters.calculix import CalculiXAdapter
from dispatch.adapters.openfoam import OpenFOAMAdapter, count_processor_dirs
from dispatch.adapters.registry import AdapterRegistry, build_default_registry
from dispatch.adapters.su2 import SU2Adapter, read_config
from dispatch.core.errors import AdapterError
from dispatch.core.plan import StepKind
from dispatch.core.validation import ReportBuilder

CONTROL_DICT = """\
/*--------------------------------*- C++ -*----------------------------------*/
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
// application simpleFoam;   <- commented out, must be ignored
application     {application};
startFrom       startTime;
startTime       0;
endTime         {end_time};
deltaT          0.001;
writeControl    timeStep;
writeInterval   20;
"""


def foam_case(
    root: Path, *, application: str = "icoFoam", processors: int = 0, end_time: float = 0.5
) -> Path:
    """Build a minimal but structurally valid OpenFOAM case."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "system").mkdir(exist_ok=True)
    (root / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(exist_ok=True)
    (root / "system" / "controlDict").write_text(
        CONTROL_DICT.format(application=application, end_time=end_time)
    )
    (root / "system" / "fvSchemes").touch()
    (root / "system" / "fvSolution").touch()
    (root / "constant" / "polyMesh" / "owner").write_text('FoamFile{}\nnote "nCells: 400";\n')
    for index in range(processors):
        (root / f"processor{index}" / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
        (root / f"processor{index}" / "constant" / "polyMesh" / "owner").touch()
    return root


def ctx_for(path: Path, cores: int = 1, **kwargs) -> CaseContext:
    """A context with a PATH that finds nothing, so which() checks are deterministic."""
    return CaseContext(workdir=path, cores=cores, env={"PATH": "/nonexistent"}, **kwargs)


def argvs(plan) -> list[list[str]]:
    return [list(step.argv) for step in plan.steps]


@pytest.fixture(autouse=True)
def ample_mpi_slots(monkeypatch):
    """Pin the machine's MPI slot count for every plan built in this module.

    Without this the plans below depend on how many physical cores the test machine has:
    a 20-rank plan gains an ``--oversubscribe`` on a 16-core laptop and loses it on a
    workstation. The slot-dependent behaviour is worth testing, but deliberately and in
    the tests written for it, not incidentally in every other assertion.
    """
    monkeypatch.setattr(mpi, "slots_available", lambda: 64)


# -- the OpenFOAM dictionary reader ---------------------------------------------------


def test_the_parser_ignores_commented_out_entries(tmp_path: Path) -> None:
    """The reason this is a tokeniser and not a regular expression.

    A commented-out alternative solver is common in real cases, and matching it would
    silently run the wrong one.
    """
    case = foam_case(tmp_path / "case", application="interFoam")
    assert foamdict.read_value(case / "system" / "controlDict", "application") == "interFoam"


def test_the_parser_ignores_block_comments() -> None:
    parsed = foamdict.parse("/* application wrongFoam; */\napplication rightFoam;")
    assert parsed["application"] == "rightFoam"


def test_the_parser_reads_nested_blocks() -> None:
    parsed = foamdict.parse("method hierarchical;\ncoeffs\n{\n    n (3 3 1);\n}\n")
    assert parsed["method"] == "hierarchical"
    assert isinstance(parsed["coeffs"], dict)


def test_set_value_preserves_the_rest_of_the_file(tmp_path: Path) -> None:
    """A user's comments and formatting must survive Dispatch editing one entry."""
    path = tmp_path / "controlDict"
    path.write_text("// a comment\napplication  icoFoam;\nendTime  10;\n// trailing\n")
    assert foamdict.set_value(path, "endTime", "20")
    text = path.read_text()
    assert "endTime  20;" in text
    assert "// a comment" in text
    assert "// trailing" in text
    assert "application  icoFoam;" in text


def test_set_value_appends_a_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "controlDict"
    path.write_text("application icoFoam;\n")
    assert foamdict.set_value(path, "stopAt", "writeNow")
    assert "stopAt writeNow;" in path.read_text()


def test_balanced_factors_multiply_to_the_count() -> None:
    for count in range(1, 65):
        x, y, z = foamdict.balanced_factors(count)
        assert x * y * z == count


def test_geometric_coefficients_are_rewritten(tmp_path: Path) -> None:
    """The bug a real tutorial case exposed: a stale ``n`` makes decomposePar refuse."""
    path = tmp_path / "decomposeParDict"
    path.write_text("numberOfSubdomains 9;\nmethod hierarchical;\ncoeffs\n{\n    n   (3 3 1);\n}\n")
    assert foamdict.set_decomposition(path, 4)
    text = path.read_text()
    assert "numberOfSubdomains 4;" in text
    assert "n   (2 2 1);" in text


def test_scotch_needs_no_coefficient_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "decomposeParDict"
    path.write_text("numberOfSubdomains 9;\nmethod scotch;\n")
    assert not foamdict.set_decomposition(path, 4)
    assert "numberOfSubdomains 4;" in path.read_text()


# -- OpenFOAM detection and validation ---------------------------------------------------


def test_openfoam_is_detected_by_controldict(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", application="interFoam")
    detection = OpenFOAMAdapter.detect(case)
    assert detection is not None
    assert detection.solver == "openfoam"
    assert detection.solver_binary == "interFoam"
    assert detection.confidence == 0.95


def test_a_plain_directory_is_not_detected(tmp_path: Path) -> None:
    assert OpenFOAMAdapter.detect(tmp_path) is None


def test_processor_directories_are_counted(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", processors=8)
    assert count_processor_dirs(case) == 8


def test_non_contiguous_processor_directories_are_ignored(tmp_path: Path) -> None:
    """A gap means an interrupted decomposition; reusing it would hand mpirun a broken case."""
    case = foam_case(tmp_path / "case", processors=4)
    (case / "processor2").rename(case / "processor9")
    assert count_processor_dirs(case) == 0


def test_a_case_without_a_mesh_fails_validation(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case")
    (case / "constant" / "polyMesh" / "owner").unlink()
    report = OpenFOAMAdapter({}).validate(ctx_for(case))
    assert not report.passed
    assert any(f.code == "missing_mesh" for f in report.errors)


def test_a_case_needing_setup_is_reported_clearly(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case")
    (case / "0").rmdir()
    (case / "0.orig").mkdir()
    report = OpenFOAMAdapter({}).validate(ctx_for(case))
    assert any(f.code == "needs_setup" for f in report.errors)


def test_a_missing_solver_binary_is_an_error(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case")
    report = OpenFOAMAdapter({}).validate(ctx_for(case))
    assert any(f.code == "solver_not_found" for f in report.errors)


def test_reusing_a_matching_decomposition_is_reported(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", processors=4)
    report = OpenFOAMAdapter({}).validate(ctx_for(case, cores=4))
    assert any("reusing" in f.message for f in report.findings)


# -- OpenFOAM planning: the decomposition table --------------------------------------------


def test_serial_case_with_no_decomposition(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", application="icoFoam")
    plan = OpenFOAMAdapter({}).plan(ctx_for(case, cores=1))
    assert argvs(plan) == [["icoFoam"]]
    assert plan.solve.kind is StepKind.SOLVE


def test_parallel_case_reuses_a_matching_decomposition(tmp_path: Path) -> None:
    """No reconstruct, no re-decompose: the existing split is already right."""
    case = foam_case(tmp_path / "case", application="interFoam", processors=4)
    plan = OpenFOAMAdapter({}).plan(ctx_for(case, cores=4))
    assert argvs(plan) == [["mpirun", "-np", "4", "interFoam", "-parallel"]]


def test_a_mismatched_decomposition_is_rebuilt(tmp_path: Path) -> None:
    """The headline feature: the user never types reconstructPar."""
    case = foam_case(tmp_path / "case", application="interFoam", processors=8)
    plan = OpenFOAMAdapter({}).plan(ctx_for(case, cores=20))
    assert argvs(plan) == [
        ["reconstructPar", "-latestTime"],
        ["rm", "-rf", *[f"processor{i}" for i in range(8)]],
        ["decomposePar", "-force"],
        ["mpirun", "-np", "20", "interFoam", "-parallel"],
    ]


def test_an_undecomposed_case_is_decomposed(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", application="pimpleFoam")
    plan = OpenFOAMAdapter({}).plan(ctx_for(case, cores=16))
    assert argvs(plan) == [
        ["decomposePar", "-force"],
        ["mpirun", "-np", "16", "pimpleFoam", "-parallel"],
    ]


def test_going_back_to_serial_reconstructs_first(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", application="icoFoam", processors=4)
    plan = OpenFOAMAdapter({}).plan(ctx_for(case, cores=1))
    assert argvs(plan) == [["reconstructPar", "-latestTime"], ["icoFoam"]]


def test_the_plan_never_reconstructs_after_the_solve(tmp_path: Path) -> None:
    """Explicitly required, and correct: reconstruction can take longer than the run."""
    case = foam_case(tmp_path / "case", processors=8)
    plan = OpenFOAMAdapter({}).plan(ctx_for(case, cores=4))
    assert plan.cleanup == ()
    assert plan.steps[-1].kind is StepKind.SOLVE


def test_reconstruct_failure_only_warns(tmp_path: Path) -> None:
    """A case decomposed but never run has nothing to reconstruct; losing the job for
    that would be wrong."""
    case = foam_case(tmp_path / "case", processors=8)
    plan = OpenFOAMAdapter({}).plan(ctx_for(case, cores=4))
    from dispatch.core.plan import FailureAction

    assert plan.steps[0].on_failure is FailureAction.WARN


def test_planning_writes_a_decompose_dict_when_absent(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case")
    OpenFOAMAdapter({}).plan(ctx_for(case, cores=6))
    written = case / "system" / "decomposeParDict"
    assert written.is_file()
    assert foamdict.read_value(written, "numberOfSubdomains") == "6"


def test_planning_preserves_an_existing_decompose_method(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case")
    path = case / "system" / "decomposeParDict"
    path.write_text("numberOfSubdomains 2;\nmethod simple;\ncoeffs { n (2 1 1); }\n")
    OpenFOAMAdapter({}).plan(ctx_for(case, cores=8))
    assert foamdict.read_value(path, "method") == "simple"
    assert foamdict.read_value(path, "numberOfSubdomains") == "8"


# -- OpenFOAM metadata and progress ------------------------------------------------------------


def test_openfoam_metadata_is_extracted(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", application="interFoam", end_time=600)
    metadata = OpenFOAMAdapter({}).collect_metadata(ctx_for(case))
    assert metadata.case["application"] == "interFoam"
    assert metadata.case["endTime"] == 600.0
    assert metadata.case["mesh_cells"] == 400


def test_openfoam_progress_is_read_from_the_log(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case", end_time=100)
    progress = OpenFOAMAdapter({}).parse_progress(
        "Time = 10\nsome output\nTime = 25\nmore output\n", ctx_for(case)
    )
    assert progress is not None
    assert progress.current == 25.0
    assert progress.fraction == pytest.approx(0.25)


def test_progress_is_none_when_the_log_says_nothing(tmp_path: Path) -> None:
    case = foam_case(tmp_path / "case")
    assert OpenFOAMAdapter({}).parse_progress("Creating mesh...\n", ctx_for(case)) is None


def test_graceful_stop_sets_stop_at_write_now(tmp_path: Path) -> None:
    """The correct way to stop a Foam run: it writes, then exits."""
    case = foam_case(tmp_path / "case")
    assert OpenFOAMAdapter({}).stop_gracefully(ctx_for(case))
    assert foamdict.read_value(case / "system" / "controlDict", "stopAt") == "writeNow"


def test_finalize_restores_stop_at_after_a_cancellation(tmp_path: Path) -> None:
    """The edit must not outlive the run it was made for.

    Left behind, ``stopAt writeNow`` makes every later run of the case write once and exit
    at its first time step -- with status 0, so nothing anywhere reports a problem and the
    case simply appears to stop working.
    """
    case = foam_case(tmp_path / "case")
    control = case / "system" / "controlDict"
    foamdict.set_value(control, "stopAt", "endTime")
    adapter = OpenFOAMAdapter({})
    ctx = ctx_for(case)

    adapter.stop_gracefully(ctx)
    assert foamdict.read_value(control, "stopAt") == "writeNow"

    adapter.finalize(ctx)
    assert foamdict.read_value(control, "stopAt") == "endTime"


def test_finalize_leaves_an_uncancelled_case_alone(tmp_path: Path) -> None:
    """No cancellation means nothing to undo, and nothing to rewrite."""
    case = foam_case(tmp_path / "case")
    control = case / "system" / "controlDict"
    foamdict.set_value(control, "stopAt", "nextWrite")
    before = control.read_text()

    OpenFOAMAdapter({}).finalize(ctx_for(case))
    assert control.read_text() == before


def test_planning_heals_a_case_a_crash_left_stopping(tmp_path: Path) -> None:
    """A daemon killed between the cancel and the restore must not poison the case forever."""
    case = foam_case(tmp_path / "case")
    control = case / "system" / "controlDict"
    foamdict.set_value(control, "stopAt", "endTime")
    adapter = OpenFOAMAdapter({})

    adapter.stop_gracefully(ctx_for(case))  # crash here: no finalize ever runs
    assert foamdict.read_value(control, "stopAt") == "writeNow"

    adapter.plan(ctx_for(case, cores=1))
    assert foamdict.read_value(control, "stopAt") == "endTime"


# -- MPI slots ------------------------------------------------------------------------------------


def test_more_ranks_than_slots_is_launched_oversubscribed() -> None:
    """The bug that made every large OpenFOAM job fail one second after starting.

    Dispatch used to admit against the *logical* CPU count while MPI sizes its slots by
    *physical* cores, so a job sized to the machine's threads was refused by the launcher
    before the solver ran at all.
    """
    argv = mpi.launch_argv(24, "interFoam", "-parallel", slots=16)
    assert argv == ["mpirun", "-np", "24", "--oversubscribe", "interFoam", "-parallel"]


def test_a_job_that_fits_is_not_oversubscribed() -> None:
    """The flag also silences the launcher's guard, so it is not added speculatively."""
    argv = mpi.launch_argv(16, "interFoam", "-parallel", slots=16)
    assert argv == ["mpirun", "-np", "16", "interFoam", "-parallel"]


def test_an_undetectable_slot_count_does_not_add_the_flag(monkeypatch) -> None:
    """Not knowing the machine is a reason to leave the launcher's own defaults alone."""
    monkeypatch.setattr(mpi, "slots_available", lambda: None)
    assert mpi.launch_argv(99, "solver") == ["mpirun", "-np", "99", "solver"]


def test_oversubscription_is_reported_at_validation(tmp_path: Path) -> None:
    builder = ReportBuilder()
    mpi.check_slots(ctx_for(tmp_path, cores=24), builder, slots=16)
    assert [f.code for f in builder.build()] == ["mpi_oversubscribed"]


def test_a_fitting_job_validates_without_a_slot_warning(tmp_path: Path) -> None:
    builder = ReportBuilder()
    mpi.check_slots(ctx_for(tmp_path, cores=8), builder, slots=16)
    assert not builder.build().findings


# -- explaining failures ---------------------------------------------------------------------------


def test_a_foam_fatal_error_is_quoted_back(tmp_path: Path) -> None:
    """Exit code 1 says nothing; the banner underneath it says everything."""
    log = (
        "Courant Number mean: 0.1 max: 0.4\n"
        "Time = 0.05\n\n"
        "--> FOAM FATAL ERROR: (openfoam-2406)\n"
        "Maximum number of iterations exceeded: 1000\n\n"
        "    From static Foam::label Foam::BisectionRoot at line 78.\n"
        "[stack trace]\n"
        "#1  Foam::error::printStack(Foam::Ostream&)\n"
    )
    summary = OpenFOAMAdapter({}).explain_failure(log, ctx_for(tmp_path / "case"))
    assert summary is not None
    assert "FOAM FATAL ERROR" in summary
    assert "Maximum number of iterations exceeded: 1000" in summary
    assert "printStack" not in summary  # the stack trace explains nothing


def test_an_mpi_launch_failure_is_explained(tmp_path: Path) -> None:
    """The exact failure from the bug report: nothing on stdout, everything on stderr."""
    log = (
        "-" * 74 + "\n"
        "There are not enough slots available in the system to satisfy the 23\n"
        "slots that were requested by the application:\n\n"
        "  interFoam\n\n"
        "Either request fewer slots for your application, or make more slots\n"
        "available for use.\n" + "-" * 74 + "\n"
    )
    summary = OpenFOAMAdapter({}).explain_failure(log, ctx_for(tmp_path / "case"))
    assert summary is not None
    assert "not enough slots" in summary
    assert not summary.startswith("-")  # the rule-off lines are noise


def test_a_parallel_fatal_error_survives_its_rank_labels(tmp_path: Path) -> None:
    """Taken verbatim from a real four-rank run, interleaving and all.

    Every rank hits the same error at the same moment and ``mpirun`` splices their output
    together mid-line, so an anchored pattern matches nothing and a naive reading repeats
    the same sentence once per rank.
    """
    log = (
        "[1] \n"
        "[1] --> FOAM FATAL ERROR: (openfoam-2406 patch=[2] \n"
        "[2] \n"
        "[2] --> FOAM FATAL ERROR: (openfoam-2406 patch=260127)\n"
        '[2] cannot find file "/cases/dam/processor2/0/p_rgh"\n'
        "[2] \n"
        "[2]     From virtual Foam::autoPtr<Foam::ISstream> Foam::readStream(...) const\n"
        "[2]     in file global/fileOperations/uncollatedFileOperation.C at line 629.\n"
        "FOAM parallel run exiting\n"
    )
    summary = OpenFOAMAdapter({}).explain_failure(log, ctx_for(tmp_path / "case"))
    assert summary is not None
    assert 'cannot find file "/cases/dam/processor2/0/p_rgh"' in summary
    assert summary.count("FOAM FATAL ERROR") == 1  # once, not once per rank
    assert "[2]" not in summary  # rank labels are not part of the explanation
    assert "uncollatedFileOperation" not in summary  # nor is the C++ it was raised from


def test_a_healthy_log_is_never_mined_for_a_scapegoat() -> None:
    """The trap this heuristic has to avoid.

    A working OpenFOAM run prints ``time step continuity errors`` on every time step. A
    marker pattern loose enough to match it would name a routine diagnostic as the cause of
    death for any job whose real error scrolled out of the tail.
    """
    healthy = (
        "Time = 0.5\n"
        "DICPCG:  Solving for p_rgh, Initial residual = 0.033, Final residual = 0.0015\n"
        "time step continuity errors : sum local = 0.0017, global = -7.1e-05\n"
        "ExecutionTime = 2.9 s  ClockTime = 3 s\n"
    )
    summary = generic_failure_summary(healthy)
    assert summary is not None
    # The last lines, in order -- not a line plucked out because it contained "errors".
    assert summary.splitlines()[-1].startswith("ExecutionTime")


def test_a_silent_log_explains_nothing_rather_than_guessing(tmp_path: Path) -> None:
    assert OpenFOAMAdapter({}).explain_failure("\n\n" + "-" * 20 + "\n", ctx_for(tmp_path)) is None


def test_the_generic_summary_falls_back_to_the_last_lines() -> None:
    """Most solvers have no recognisable error format; the end of the log is still the answer."""
    summary = generic_failure_summary("setting up\nsolving\nsomething went sideways\n")
    assert summary is not None
    assert "something went sideways" in summary


# -- SU2 ------------------------------------------------------------------------------------------

SU2_CONFIG = """\
% SU2 configuration
SOLVER= RANS
MATH_PROBLEM= DIRECT
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
ITER= 500
MACH_NUMBER= 0.8
AOA= 3.06
RESTART_SOL= NO
"""


def su2_case(root: Path, *, name: str = "config.cfg", mesh: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(SU2_CONFIG)
    if mesh:
        (root / "mesh.su2").write_text("NDIME= 2\n")
    return root


def test_su2_is_detected_by_its_config(tmp_path: Path) -> None:
    case = su2_case(tmp_path / "wing")
    detection = SU2Adapter.detect(case)
    assert detection is not None
    assert detection.solver == "su2"
    assert detection.confidence == 0.85
    assert detection.entry is not None and detection.entry.name == "config.cfg"


def test_a_cfg_without_su2_keys_is_not_detected(tmp_path: Path) -> None:
    """`.cfg` is a common extension; the content has to say SU2."""
    (tmp_path / "settings.cfg").write_text("[section]\nkey = value\n")
    assert SU2Adapter.detect(tmp_path) is None


def test_su2_config_parsing() -> None:
    settings = read_config(Path("/dev/null"))
    assert settings == {}


def test_su2_plans_serial_and_parallel(tmp_path: Path) -> None:
    """SU2 partitions internally, so there is nothing to prepare."""
    case = su2_case(tmp_path / "wing")
    adapter = SU2Adapter({})
    assert argvs(adapter.plan(ctx_for(case, cores=1))) == [["SU2_CFD", "config.cfg"]]
    assert argvs(adapter.plan(ctx_for(case, cores=8))) == [
        ["mpirun", "-np", "8", "SU2_CFD", "config.cfg"]
    ]


def test_su2_reports_a_missing_mesh(tmp_path: Path) -> None:
    case = su2_case(tmp_path / "wing", mesh=False)
    report = SU2Adapter({}).validate(ctx_for(case))
    assert any(f.code == "missing_mesh" for f in report.errors)


def test_su2_warns_about_a_restart_with_no_restart_file(tmp_path: Path) -> None:
    """A classic silent failure worth catching before the job is queued."""
    case = su2_case(tmp_path / "wing")
    (case / "config.cfg").write_text(SU2_CONFIG.replace("RESTART_SOL= NO", "RESTART_SOL= YES"))
    report = SU2Adapter({}).validate(ctx_for(case))
    assert any(f.code == "missing_restart" for f in report.findings)


def test_su2_lists_other_configs_rather_than_guessing(tmp_path: Path) -> None:
    case = su2_case(tmp_path / "wing")
    (case / "adjoint.cfg").write_text(SU2_CONFIG.replace("DIRECT", "CONTINUOUS_ADJOINT"))
    report = SU2Adapter({}).validate(ctx_for(case))
    assert any("adjoint.cfg" in f.message for f in report.findings)


def test_su2_uses_the_config_detection_chose(tmp_path: Path) -> None:
    """A directory with two configs must run the one the user picked."""
    case = su2_case(tmp_path / "wing")
    (case / "adjoint.cfg").write_text(SU2_CONFIG)
    plan = SU2Adapter({}).plan(ctx_for(case, cores=1, entry=case / "adjoint.cfg"))
    assert argvs(plan) == [["SU2_CFD", "adjoint.cfg"]]


def test_su2_metadata(tmp_path: Path) -> None:
    case = su2_case(tmp_path / "wing")
    metadata = SU2Adapter({}).collect_metadata(ctx_for(case))
    assert metadata.case["solver_type"] == "RANS"
    assert metadata.case["mach"] == 0.8
    assert metadata.case["aoa"] == 3.06
    assert metadata.case["iterations"] == 500


# -- Basilisk --------------------------------------------------------------------------------------

BASILISK_SOURCE = """\
#include "navier-stokes/centered.h"
#include "two-phase.h"

int main() {
  run();
}
"""


def basilisk_case(root: Path, *, name: str = "bubble.c") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(BASILISK_SOURCE)
    return root


def test_basilisk_is_detected_with_lower_confidence(tmp_path: Path) -> None:
    """A `.c` file is genuinely weaker evidence than system/controlDict."""
    case = basilisk_case(tmp_path / "bubble")
    detection = BasiliskAdapter.detect(case)
    assert detection is not None
    assert detection.confidence == 0.70
    assert detection.confidence < 0.95


def test_a_plain_c_file_is_not_a_basilisk_project(tmp_path: Path) -> None:
    (tmp_path / "main.c").write_text("#include <stdio.h>\nint main(){return 0;}\n")
    assert BasiliskAdapter.detect(tmp_path) is None


def test_basilisk_compiles_then_runs(tmp_path: Path) -> None:
    """Compilation is an ordinary preparation step.

    The scheduler needed no changes to support a solver that must be built first, which is
    the concrete evidence that the adapter boundary is in the right place.
    """
    case = basilisk_case(tmp_path / "bubble")
    plan = BasiliskAdapter({}).plan(ctx_for(case, cores=1))
    assert argvs(plan) == [
        ["qcc", "-O2", "-Wall", "-o", "bubble", "bubble.c", "-lm"],
        ["./bubble"],
    ]
    assert plan.steps[0].kind is StepKind.PREPARE
    assert plan.steps[1].kind is StepKind.SOLVE


def test_basilisk_adds_mpi_flags_when_parallel(tmp_path: Path) -> None:
    """`-D_MPI=N` must match the rank count the job actually launches with."""
    case = basilisk_case(tmp_path / "bubble")
    plan = BasiliskAdapter({}).plan(ctx_for(case, cores=8))
    assert "-D_MPI=8" in plan.steps[0].argv
    assert list(plan.solve.argv) == ["mpirun", "-np", "8", "./bubble"]


def test_basilisk_refuses_to_guess_between_two_sources(tmp_path: Path) -> None:
    case = basilisk_case(tmp_path / "project")
    (case / "other.c").write_text(BASILISK_SOURCE)
    report = BasiliskAdapter({}).validate(ctx_for(case))
    assert any(f.code == "ambiguous_source" for f in report.errors)


def test_basilisk_skips_qcc_generated_sources(tmp_path: Path) -> None:
    """Otherwise a previous build's output looks like another candidate source."""
    case = basilisk_case(tmp_path / "bubble")
    (case / "_bubble.c").write_text(BASILISK_SOURCE)
    report = BasiliskAdapter({}).validate(ctx_for(case))
    assert not any(f.code == "ambiguous_source" for f in report.findings)


def test_basilisk_progress_is_optional(tmp_path: Path) -> None:
    """Basilisk output is user-defined; None is a perfectly good answer."""
    case = basilisk_case(tmp_path / "bubble")
    adapter = BasiliskAdapter({})
    assert adapter.parse_progress("nothing useful\n", ctx_for(case)) is None
    progress = adapter.parse_progress("t = 1.25 dt = 0.01\n", ctx_for(case))
    assert progress is not None and progress.current == 1.25
    assert progress.fraction is None


# -- CalculiX --------------------------------------------------------------------------------------

DECK = """\
*HEADING
A simple static analysis
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
*ELEMENT, TYPE=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
*STEP
*STATIC
*END STEP
"""


def test_calculix_is_detected_by_a_deck(tmp_path: Path) -> None:
    (tmp_path / "beam.inp").write_text(DECK)
    detection = CalculiXAdapter.detect(tmp_path)
    assert detection is not None
    assert detection.solver == "calculix"
    assert detection.solver_binary == "ccx"


def test_an_inp_without_step_is_not_a_deck(tmp_path: Path) -> None:
    (tmp_path / "mesh.inp").write_text("*NODE\n1, 0.0, 0.0, 0.0\n")
    assert CalculiXAdapter.detect(tmp_path) is None


def test_calculix_uses_threads_not_ranks(tmp_path: Path) -> None:
    """The case that would have been awkward if the interface had assumed MPI."""
    (tmp_path / "beam.inp").write_text(DECK)
    plan = CalculiXAdapter({}).plan(ctx_for(tmp_path, cores=8))
    assert argvs(plan) == [["ccx", "-i", "beam"]]
    assert plan.solve.env is not None
    assert plan.solve.env["OMP_NUM_THREADS"] == "8"


def test_calculix_strips_the_extension(tmp_path: Path) -> None:
    """Passing the full filename is the most common CalculiX mistake."""
    (tmp_path / "beam.inp").write_text(DECK)
    plan = CalculiXAdapter({}).plan(ctx_for(tmp_path))
    assert "beam" in plan.solve.argv and "beam.inp" not in plan.solve.argv


def test_calculix_reports_a_missing_include(tmp_path: Path) -> None:
    (tmp_path / "beam.inp").write_text(DECK + "*INCLUDE, INPUT=mesh.inp\n")
    report = CalculiXAdapter({}).validate(ctx_for(tmp_path))
    assert any(f.code == "missing_include" for f in report.errors)


def test_calculix_metadata(tmp_path: Path) -> None:
    (tmp_path / "beam.inp").write_text(DECK)
    metadata = CalculiXAdapter({}).collect_metadata(ctx_for(tmp_path, cores=4))
    assert metadata.case["analysis"] == "Static"
    assert metadata.case["steps"] == 1
    assert metadata.case["threads"] == 4


# -- the registry ----------------------------------------------------------------------------------


def test_every_builtin_adapter_registers() -> None:
    registry = build_default_registry({}, load_plugins=False)
    assert set(registry.names) == {"openfoam", "su2", "basilisk", "calculix", "ml", "pinn"}
    assert registry.rejected == []


def test_an_adapter_with_the_wrong_api_version_is_refused() -> None:
    """And the daemon still starts, with the remaining adapters."""

    class Ancient(BaseAdapter):
        api_version = ADAPTER_API_VERSION + 99
        name = "ancient"

        @classmethod
        def detect(cls, path: Path):
            return None

        def validate(self, ctx):
            raise NotImplementedError

        def plan(self, ctx):
            raise NotImplementedError

    registry = AdapterRegistry({})
    assert not registry.register(Ancient)
    assert registry.rejected[0].name == "ancient"
    assert "version" in registry.rejected[0].reason
    assert registry.names == ()


def test_registering_the_same_name_twice_is_refused() -> None:
    registry = AdapterRegistry({})
    assert registry.register(OpenFOAMAdapter)
    assert not registry.register(OpenFOAMAdapter)


def test_an_unknown_adapter_name_lists_the_known_ones() -> None:
    registry = build_default_registry({}, load_plugins=False)
    with pytest.raises(AdapterError) as excinfo:
        registry.get("nonexistent")
    assert "openfoam" in str(excinfo.value)


def test_detection_ranks_by_confidence(tmp_path: Path) -> None:
    """A directory that is both a Foam case and holds a `.c` file resolves to Foam."""
    case = foam_case(tmp_path / "hybrid")
    (case / "post.c").write_text(BASILISK_SOURCE)
    registry = build_default_registry({}, load_plugins=False)
    detections = registry.detect(case)
    assert [d.solver for d in detections] == ["openfoam", "basilisk"]
    best = registry.best_detection(case)
    assert best is not None and best.solver == "openfoam"


def test_an_adapter_that_raises_during_detection_is_skipped(tmp_path: Path) -> None:
    """One broken adapter must not make browsing impossible."""

    class Exploding(BaseAdapter):
        name = "exploding"

        @classmethod
        def detect(cls, path: Path):
            raise RuntimeError("boom")

        def validate(self, ctx):
            raise NotImplementedError

        def plan(self, ctx):
            raise NotImplementedError

    registry = AdapterRegistry({})
    registry.register(Exploding)
    registry.register(OpenFOAMAdapter)
    case = foam_case(tmp_path / "case")
    assert [d.solver for d in registry.detect(case)] == ["openfoam"]


def test_adapter_settings_reach_the_adapter() -> None:
    registry = AdapterRegistry({"openfoam": {"bashrc": "/custom/bashrc"}})
    registry.register(OpenFOAMAdapter)
    adapter = registry.get("openfoam")
    assert isinstance(adapter, BaseAdapter)
    assert adapter.setting("bashrc") == "/custom/bashrc"


def test_adapter_instances_are_cached() -> None:
    """Adapters memoise expensive environment probes; rebuilding would discard that."""
    registry = build_default_registry({}, load_plugins=False)
    assert registry.get("openfoam") is registry.get("openfoam")


def test_every_adapter_declares_a_metadata_spec() -> None:
    for adapter in build_default_registry({}, load_plugins=False):
        spec = adapter.metadata_spec
        assert spec.ref.adapter == adapter.name
        assert spec.fields, f"{adapter.name} declares no metadata fields"


# -- sweep folders ---------------------------------------------------------------------------
#
# Detection has to be conservative: missing a sweep costs the user the convenience they had
# before it existed, while inventing one queues a pile of jobs nobody asked for.


def sweep_root(tmp_path: Path, *names: str) -> Path:
    """A directory containing OpenFOAM cases under the given names."""
    root = tmp_path / "sweep"
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        foam_case(root / name, application="icoFoam")
    return root


def test_a_folder_of_same_solver_cases_is_a_sweep(tmp_path: Path) -> None:
    registry = build_default_registry()
    detection = registry.detect_sweep(sweep_root(tmp_path, "case_001", "case_002", "case_003"))

    assert detection is not None
    assert detection.solver == "openfoam"
    assert detection.count == 3


def test_sweep_cases_are_ordered_naturally(tmp_path: Path) -> None:
    """``case_10`` sorts after ``case_9``, not between ``case_1`` and ``case_2``.

    Purely lexical ordering is deterministic but reads as a bug the first time a sweep has
    ten members, and the order cases run in is the order the user will compare results in.
    """
    root = sweep_root(tmp_path, "case_1", "case_2", "case_9", "case_10", "case_20")
    detection = build_default_registry().detect_sweep(root)

    assert detection is not None
    assert [c.name for c in detection.cases] == [
        "case_1",
        "case_2",
        "case_9",
        "case_10",
        "case_20",
    ]


def test_a_directory_that_is_itself_a_case_is_not_a_sweep(tmp_path: Path) -> None:
    """A case containing cases is a case. Submitting the parent must run the parent."""
    root = sweep_root(tmp_path, "case_001", "case_002")
    foam_case(root, application="interFoam")  # make the parent a case too

    assert build_default_registry().detect_sweep(root) is None


def test_one_unrecognised_sibling_disqualifies_a_sweep(tmp_path: Path) -> None:
    """The conservative rule that stops a project directory being submitted wholesale."""
    root = sweep_root(tmp_path, "case_001", "case_002")
    (root / "scripts").mkdir()

    assert build_default_registry().detect_sweep(root) is None


def test_a_mixed_solver_folder_is_not_a_sweep(tmp_path: Path) -> None:
    """Two solvers is a project, not a sweep: there is no single cores-per-job to set."""
    root = sweep_root(tmp_path, "case_001")
    su2_case(root / "case_002")

    assert build_default_registry().detect_sweep(root) is None


def test_a_single_case_is_not_a_sweep(tmp_path: Path) -> None:
    """A sweep of one is a job. Offering the sweep flow for it is noise."""
    assert build_default_registry().detect_sweep(sweep_root(tmp_path, "only")) is None


def test_an_empty_directory_is_not_a_sweep(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build_default_registry().detect_sweep(empty) is None


# -- resuming from a case's own saved state ----------------------------------------------------


def test_a_case_that_has_written_nothing_has_no_resume_point(tmp_path: Path) -> None:
    """A run interrupted before its first write resumes from the beginning, and says so."""
    case = foam_case(tmp_path / "fresh", application="interFoam")
    assert OpenFOAMAdapter({}).resume_point(ctx_for(case)) is None


def test_the_resume_point_is_read_from_the_cases_written_times(tmp_path: Path) -> None:
    """The restart point comes off the disk, never from anything Dispatch recorded."""
    case = foam_case(tmp_path / "written", application="interFoam")
    for time_dir in ("0.5", "7500", "250"):
        (case / time_dir).mkdir()

    assert OpenFOAMAdapter({}).resume_point(ctx_for(case)) == "t = 7500"


def test_a_decomposed_case_resumes_from_its_processor_times(tmp_path: Path) -> None:
    """A parallel run writes inside ``processorN``, which is where a real sweep job resumes."""
    case = foam_case(tmp_path / "parallel", application="interFoam", processors=4)
    (case / "processor0" / "1200").mkdir(parents=True)
    (case / "processor1" / "1200").mkdir(parents=True)

    assert OpenFOAMAdapter({}).resume_point(ctx_for(case)) == "t = 1200"


def test_the_initial_condition_is_not_a_resume_point(tmp_path: Path) -> None:
    """``0`` is where the case starts, so reporting it would claim preserved work."""
    case = foam_case(tmp_path / "zero-only", application="interFoam")
    (case / "0").mkdir(exist_ok=True)

    assert OpenFOAMAdapter({}).resume_point(ctx_for(case)) is None


def test_resuming_points_the_case_at_its_latest_time(tmp_path: Path) -> None:
    """OpenFOAM's own restart mechanism, driven by the flag rather than reinvented."""
    case = foam_case(tmp_path / "resume", application="interFoam")
    (case / "300").mkdir()
    control = case / "system" / "controlDict"
    foamdict.set_value(control, "startFrom", "startTime")

    adapter = OpenFOAMAdapter({})
    adapter.plan(ctx_for(case, cores=1, resume=True))

    assert foamdict.read_value(control, "startFrom") == "latestTime"


def test_the_resume_edit_does_not_outlive_the_run(tmp_path: Path) -> None:
    """A `startFrom` left behind would silently change where every later run begins."""
    case = foam_case(tmp_path / "restored", application="interFoam")
    (case / "300").mkdir()
    control = case / "system" / "controlDict"
    foamdict.set_value(control, "startFrom", "startTime")

    adapter = OpenFOAMAdapter({})
    ctx = ctx_for(case, cores=1, resume=True)
    adapter.plan(ctx)
    adapter.finalize(ctx)

    assert foamdict.read_value(control, "startFrom") == "startTime"


def test_an_ordinary_run_never_touches_start_from(tmp_path: Path) -> None:
    """The whole mechanism is inert unless a resume was actually asked for."""
    case = foam_case(tmp_path / "ordinary", application="interFoam")
    control = case / "system" / "controlDict"
    foamdict.set_value(control, "startFrom", "startTime")

    adapter = OpenFOAMAdapter({})
    adapter.plan(ctx_for(case, cores=1))

    assert foamdict.read_value(control, "startFrom") == "startTime"
    assert not (case / "system" / ".dispatch-startFrom").exists()
