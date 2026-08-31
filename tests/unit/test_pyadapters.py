"""The ML and PINN adapters.

Runnable on a machine with no PyTorch, no TensorFlow, no JAX, no DeepXDE, and no GPU:
every case here is a directory of text files, and every assertion is about a command list,
an environment, or a detection decision.

The bulk of it is about detection *refusing*. An adapter that claims every Python project
on the machine would make the submit wizard useless, so most of these tests describe
directories that must not be claimed.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from dispatch.adapters.base import CaseContext
from dispatch.adapters.gpuenv import GPU_VISIBILITY_VARS
from dispatch.adapters.ml import MLAdapter
from dispatch.adapters.pinn import PINNAdapter
from dispatch.adapters.pyjob import DECLARATION_FILE, parse_metric_log, read_declaration
from dispatch.adapters.registry import build_default_registry


def project(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(body))
    return path


def ctx(path: Path, *, cores: int = 4, gpus: int = 0) -> CaseContext:
    """A context with a PATH that finds nothing, so ``which`` is deterministic."""
    return CaseContext(workdir=path, cores=cores, gpus=gpus, env={"PATH": "/nonexistent"})


def argv(adapter, context) -> list[str]:
    return list(adapter.plan(context).solve.argv)


# -- ML detection: mostly about refusing ---------------------------------------------------


def test_a_plain_python_project_is_not_a_training_run(tmp_path: Path) -> None:
    """The single most important test in this file."""
    case = project(tmp_path / "lib", {"pyproject.toml": "[project]\nname='x'\n", "main.py": ""})
    assert MLAdapter.detect(case) is None


def test_a_bare_script_is_not_a_project(tmp_path: Path) -> None:
    """``train.py`` in a scratch folder is not something somebody set up to be scheduled."""
    assert MLAdapter.detect(project(tmp_path / "scratch", {"train.py": ""})) is None


def test_an_empty_directory_is_not_claimed(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert MLAdapter.detect(tmp_path / "empty") is None
    assert PINNAdapter.detect(tmp_path / "empty") is None


def test_requirements_alone_are_not_evidence(tmp_path: Path) -> None:
    case = project(tmp_path / "deps", {"requirements.txt": "torch\nnumpy\n"})
    assert MLAdapter.detect(case) is None


def test_importing_torch_is_not_evidence(tmp_path: Path) -> None:
    """Plenty of libraries import torch. It says nothing about being a job."""
    case = project(
        tmp_path / "lib", {"pyproject.toml": "[project]\n", "model.py": "import torch\n"}
    )
    assert MLAdapter.detect(case) is None


def test_a_training_script_in_a_real_project_is_claimed(tmp_path: Path) -> None:
    case = project(
        tmp_path / "run", {"pyproject.toml": "[project]\nname='r'\n", "train.py": "import torch\n"}
    )
    detection = MLAdapter.detect(case)
    assert detection is not None
    assert detection.solver == "ml"
    assert detection.confidence == pytest.approx(0.6)
    assert detection.entry == case / "train.py"


# -- PINN detection ----------------------------------------------------------------------


def test_a_pinn_library_import_is_evidence(tmp_path: Path) -> None:
    """``import deepxde`` is not something a project does by accident."""
    case = project(
        tmp_path / "burgers",
        {"requirements.txt": "deepxde\n", "train.py": "import deepxde as dde\n"},
    )
    detection = PINNAdapter.detect(case)
    assert detection is not None
    assert detection.confidence == pytest.approx(0.75)
    assert detection.detail["library"] == "deepxde"


def test_a_pinn_project_outranks_the_generic_ml_adapter(tmp_path: Path) -> None:
    """Both claim it; detection must resolve without asking the user."""
    case = project(
        tmp_path / "burgers",
        {
            "pyproject.toml": "[project]\ndependencies=['deepxde']\n",
            "train.py": "import deepxde\n",
        },
    )
    registry = build_default_registry({}, load_plugins=False)
    best = registry.best_detection(case)
    assert best is not None and best.solver == "pinn"


def test_a_directory_named_pinn_is_not_evidence(tmp_path: Path) -> None:
    """Naming is a fragile heuristic; the declaration file exists instead."""
    case = project(tmp_path / "pinn_solver", {"pyproject.toml": "[project]\n", "run.py": ""})
    assert PINNAdapter.detect(case) is None


def test_a_pinn_library_with_nothing_runnable_is_not_claimed(tmp_path: Path) -> None:
    """Claiming it would produce a job that fails validation for no actionable reason."""
    case = project(tmp_path / "notes", {"requirements.txt": "deepxde\n"})
    assert PINNAdapter.detect(case) is None


# -- the explicit declaration --------------------------------------------------------------


def test_a_declaration_settles_the_question(tmp_path: Path) -> None:
    case = project(
        tmp_path / "case",
        {
            DECLARATION_FILE: '[job]\nadapter = "pinn"\nentrypoint = "solve.py"\n',
            "solve.py": "",
        },
    )
    detection = PINNAdapter.detect(case)
    assert detection is not None
    assert detection.confidence == pytest.approx(0.95)
    assert detection.detail["declared"] is True


def test_a_declaration_for_one_adapter_silences_the_other(tmp_path: Path) -> None:
    """The user has already answered the question, so nothing else should volunteer."""
    case = project(
        tmp_path / "case",
        {
            DECLARATION_FILE: '[job]\nadapter = "pinn"\n',
            "pyproject.toml": "[project]\n",
            "train.py": "",
        },
    )
    assert MLAdapter.detect(case) is None
    assert PINNAdapter.detect(case) is not None


def test_a_malformed_declaration_is_ignored_not_fatal(tmp_path: Path) -> None:
    """Detection runs on every directory browsed to; one bad file must not break it."""
    case = project(tmp_path / "case", {DECLARATION_FILE: "this is not = valid toml ["})
    assert read_declaration(case) is None
    assert MLAdapter.detect(case) is None


def test_a_declaration_without_a_job_table_is_ignored(tmp_path: Path) -> None:
    case = project(tmp_path / "case", {DECLARATION_FILE: '[tool.other]\nx = 1\n'})
    assert read_declaration(case) is None


def test_a_missing_declaration_is_not_an_error(tmp_path: Path) -> None:
    (tmp_path / "bare").mkdir()
    assert read_declaration(tmp_path / "bare") is None


# -- execution plans -------------------------------------------------------------------------


def test_the_plan_runs_the_declared_entrypoint_with_its_arguments(tmp_path: Path) -> None:
    """"Do not assume the script is always called train.py", made concrete."""
    case = project(
        tmp_path / "case",
        {
            DECLARATION_FILE: (
                '[job]\nadapter = "ml"\nentrypoint = "fit_operator.py"\n'
                'args = ["--config", "configs/a.yaml"]\n'
            ),
            "fit_operator.py": "",
        },
    )
    command = argv(MLAdapter({}), ctx(case))
    assert command[1:] == ["fit_operator.py", "--config", "configs/a.yaml"]


def test_the_plan_falls_back_to_a_conventional_entrypoint(tmp_path: Path) -> None:
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    assert argv(MLAdapter({}), ctx(case))[1:] == ["train.py"]


def test_a_module_entrypoint_uses_dash_m(tmp_path: Path) -> None:
    case = project(
        tmp_path / "case",
        {DECLARATION_FILE: '[job]\nadapter = "ml"\nentrypoint = "pkg.train"\nmodule = true\n'},
    )
    assert argv(MLAdapter({}), ctx(case))[1:] == ["-m", "pkg.train"]


def test_a_project_virtualenv_is_preferred(tmp_path: Path) -> None:
    """A project that ships one means it."""
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    interpreter = case / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("")
    assert argv(MLAdapter({}), ctx(case))[0] == str(interpreter)


def test_without_a_virtualenv_a_real_interpreter_is_used(tmp_path: Path) -> None:
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    assert argv(MLAdapter({}), ctx(case))[0] == sys.executable


def test_the_plan_is_a_single_solve_step(tmp_path: Path) -> None:
    """No implicit ``pip install``: a scheduler must not mutate an environment uninvited."""
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    plan = MLAdapter({}).plan(ctx(case))
    assert len(plan.steps) == 1
    assert plan.prepare == ()


def test_planning_without_an_entrypoint_fails_loudly(tmp_path: Path) -> None:
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n"})
    with pytest.raises(FileNotFoundError, match=DECLARATION_FILE):
        MLAdapter({}).plan(ctx(case))


# -- the GPU environment ---------------------------------------------------------------------


def test_a_gpu_job_is_not_told_to_hide_the_devices(tmp_path: Path) -> None:
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    env = MLAdapter({}).plan(ctx(case, gpus=1)).solve.env or {}
    assert env.get("CUDA_VISIBLE_DEVICES") is None


def test_a_cpu_job_is_told_it_has_no_devices(tmp_path: Path) -> None:
    """The ledger says it holds no GPU; the process is told the same thing."""
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    env = PINNAdapter({}).plan(ctx(case, gpus=0)).solve.env or {}
    for name in GPU_VISIBILITY_VARS:
        assert env[name] == ""


def test_an_existing_device_restriction_is_preserved_for_a_gpu_job(tmp_path: Path) -> None:
    """An administrator's machine-wide setting is not ours to overwrite."""
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    context = CaseContext(
        workdir=case, cores=1, gpus=1, env={"PATH": "/nonexistent", "CUDA_VISIBLE_DEVICES": "2"}
    )
    env = MLAdapter({}).plan(context).solve.env or {}
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_output_is_unbuffered(tmp_path: Path) -> None:
    """Otherwise the live log, the progress reader, and the plot all show nothing."""
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    env = MLAdapter({}).plan(ctx(case)).solve.env or {}
    assert env["PYTHONUNBUFFERED"] == "1"


def test_the_cfd_adapters_also_hide_gpus_from_cpu_work(tmp_path: Path) -> None:
    """The rule is machine-wide, not a property of the Python adapters."""
    from dispatch.adapters.calculix import CalculiXAdapter

    case = project(tmp_path / "deck", {"model.inp": "*STEP\n*STATIC\n*END STEP\n"})
    env = CalculiXAdapter({}).plan(ctx(case)).solve.env or {}
    assert env["CUDA_VISIBLE_DEVICES"] == ""


# -- validation ------------------------------------------------------------------------------


def test_a_project_with_no_entrypoint_fails_validation(tmp_path: Path) -> None:
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n"})
    report = MLAdapter({}).validate(ctx(case))
    assert not report.passed
    assert any(finding.code == "missing_entrypoint" for finding in report.errors)


def test_a_declared_entrypoint_that_does_not_exist_fails_validation(tmp_path: Path) -> None:
    case = project(
        tmp_path / "case", {DECLARATION_FILE: '[job]\nadapter = "ml"\nentrypoint = "gone.py"\n'}
    )
    report = MLAdapter({}).validate(ctx(case))
    assert any(finding.code == "missing_entrypoint" for finding in report.errors)


def test_a_missing_interpreter_fails_validation(tmp_path: Path) -> None:
    case = project(
        tmp_path / "case",
        {DECLARATION_FILE: '[job]\nadapter = "ml"\npython = "python-nine"\n', "train.py": ""},
    )
    report = MLAdapter({}).validate(ctx(case))
    assert any(finding.code == "missing_interpreter" for finding in report.errors)


def test_cpu_only_is_said_out_loud(tmp_path: Path) -> None:
    """A GPU job that quietly ran on CPU is much harder to notice than one that failed."""
    case = project(tmp_path / "case", {"pyproject.toml": "[project]\n", "train.py": ""})
    report = MLAdapter({}).validate(ctx(case, gpus=0))
    assert any(finding.code == "cpu_only" for finding in report.findings)
    assert report.passed, "a note, not a blocker"


# -- metadata and progress ---------------------------------------------------------------------


def test_metadata_records_what_ran_and_on_what(tmp_path: Path) -> None:
    case = project(
        tmp_path / "case",
        {"requirements.txt": "deepxde\n", "train.py": "import deepxde\n"},
    )
    metadata = PINNAdapter({}).collect_metadata(ctx(case, gpus=2))
    assert metadata.case["entrypoint"] == "train.py"
    assert metadata.case["library"] == "deepxde"
    assert metadata.case["gpus"] == 2


def test_the_pinn_adapter_suggests_useful_tags(tmp_path: Path) -> None:
    case = project(tmp_path / "case", {"requirements.txt": "deepxde\n", "train.py": ""})
    assert set(PINNAdapter({}).suggest_tags(ctx(case))) >= {"pinn", "deepxde"}


@pytest.mark.parametrize(
    ("line", "current", "total"),
    [
        ("Epoch 3/50 - loss: 0.234", 3.0, 50.0),
        ("step=1000 loss=1.2e-3", 1000.0, None),
        ("iteration: 42", 42.0, None),
    ],
)
def test_progress_is_read_from_common_training_output(
    tmp_path: Path, line: str, current: float, total: float | None
) -> None:
    progress = MLAdapter({}).parse_progress(line, ctx(tmp_path))
    assert progress is not None
    assert progress.current == current
    assert progress.total == total


def test_output_with_no_counter_reports_no_progress(tmp_path: Path) -> None:
    assert MLAdapter({}).parse_progress("loading dataset\n", ctx(tmp_path)) is None


# -- training curves ------------------------------------------------------------------------


def test_training_metrics_become_series() -> None:
    log = "\n".join(
        f"epoch {epoch}/40  loss: {0.1 / epoch:.6g}  val_loss: {0.15 / epoch:.6g}"
        for epoch in range(1, 6)
    )
    data = parse_metric_log(log)
    assert data.samples == 5
    assert {item.key for item in data.series} == {"epoch", "loss", "val_loss"}
    assert data.get("epoch") is not None and data.get("epoch").axis  # type: ignore[union-attr]


def test_a_metric_without_a_separator_is_not_invented() -> None:
    """``GPU 0`` and ``Epoch 3`` have the same shape as ``loss 0.2``; only ``:`` counts."""
    data = parse_metric_log("epoch 1\nusing GPU 0\nloss 0.5\n")
    assert "gpu" not in {item.key for item in data.series}
    assert "loss" not in {item.key for item in data.series}


def test_output_with_nothing_numerical_yields_no_series() -> None:
    assert not parse_metric_log("loading dataset\ndone\n")


def test_metrics_before_the_first_counter_are_ignored() -> None:
    """They belong to setup, not to a training step, and have no x value."""
    data = parse_metric_log("lr: 0.001\nepoch 1\nloss: 0.5\n")
    assert data.samples == 1
    assert "lr" not in {item.key for item in data.series}


def test_the_adapter_exposes_the_curve_parser(tmp_path: Path) -> None:
    data = MLAdapter({}).parse_series("epoch 1\nloss: 0.5\nepoch 2\nloss: 0.2\n", ctx(tmp_path))
    assert data.samples == 2
    assert data.get("loss") is not None
