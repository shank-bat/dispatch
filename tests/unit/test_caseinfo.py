"""The case information view's extraction layer.

Everything the ``i`` page shows is produced here, by the adapter, and the screen only lays
it out -- which is the architectural point: a second solver implements ``describe_case`` and
gets the page with no interface change. So these tests exercise the extraction, not the
rendering, and they do it against cases built from text files with no solver installed.

The 2-D/3-D and bounds tests belong to the same layer: whether a case is planar is a fact
about the solver's own conventions (an ``empty`` patch type), which is why the adapter
answers it and generic code never guesses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dispatch.adapters.base import BaseAdapter, CaseContext
from dispatch.adapters.foammesh import boundary_patches, mesh_counts, read_bounds, read_geometry
from dispatch.adapters.openfoam import OpenFOAMAdapter
from dispatch.core.caseinfo import CaseReport, InfoField, InfoSection, ReportBuilder
from dispatch.core.geometry import Bounds, CaseGeometry, Dimensionality

CONTROL = """\
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.054;
deltaT          5.4e-07;
writeControl    timeStep;
writeInterval   200;
purgeWrite      5;
writeFormat     binary;
adjustTimeStep  no;
maxCo           0.9;
"""

BOUNDARY = """\
FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
6
(
    INLET { type patch; nFaces 891; startFace 765928; }
    TOP { type patch; nFaces 458; startFace 766819; }
    OUTLET { type patch; nFaces 472; startFace 767277; }
    WING { type wall; nFaces 496; startFace 767749; }
    frontAndBackPlanes { type empty; nFaces 767320; startFace 768245; }
)
"""

BOUNDARY_3D = BOUNDARY.replace("type empty;", "type wall;")

BLOCK_MESH = """\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale   0.001;
vertices
(
    (0 0 0)
    (1000 0 0)
    (1000 500 0)
    (0 500 0)
    (0 0 20)
    (1000 0 20)
    (1000 500 20)
    (0 500 20)
);
blocks ( hex (0 1 2 3 4 5 6 7) (40 20 1) simpleGrading (1 1 1) );
"""

OWNER = (
    "FoamFile { version 2.0; format binary; class labelList; object owner; }\n"
    'note        "nPoints:770104  nCells:383660  nFaces:1536032  nInternalFaces:765928";\n'
)


def case(
    tmp_path: Path,
    *,
    boundary: str | None = BOUNDARY,
    owner: str | None = OWNER,
    block_mesh: str | None = BLOCK_MESH,
    control: str | None = CONTROL,
    fields: tuple[str, ...] = ("p", "U"),
    times: tuple[str, ...] = (),
    processors: int = 0,
) -> Path:
    root = tmp_path / "case"
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    if control is not None:
        (root / "system" / "controlDict").write_text(control)
    if boundary is not None:
        (root / "constant" / "polyMesh" / "boundary").write_text(boundary)
    if owner is not None:
        (root / "constant" / "polyMesh" / "owner").write_text(owner)
    if block_mesh is not None:
        (root / "system" / "blockMeshDict").write_text(block_mesh)
    if fields:
        (root / "0").mkdir(exist_ok=True)
        for name in fields:
            (root / "0" / name).write_text(
f"FoamFile {{ object {name}; }}\ndimensions      [0 2 -2 0 0 0 0];\n"
            )
    for time in times:
        (root / time).mkdir(exist_ok=True)
        (root / time / "U").write_text("// field\n")
    for rank in range(processors):
        (root / f"processor{rank}" / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    return root


def report(root: Path) -> CaseReport:
    return OpenFOAMAdapter({}).describe_case(CaseContext(workdir=root, cores=4))


def fields_of(result: CaseReport, title: str) -> dict[str, str]:
    section = next((item for item in result.present if item.title == title), None)
    return {} if section is None else {f.label: f.value for f in section.fields}


# -- the generic builder -------------------------------------------------------------------


def test_a_field_with_nothing_to_say_is_omitted() -> None:
    """A row reading ``End time  None`` is worse than no row."""
    builder = ReportBuilder("case", "solver")
    builder.section("time")
    builder.field("End time", None)
    builder.field("Start time", "")
    builder.field("Time step", 0.01)
    assert fields_of(builder.build(), "time") == {"Time step": "0.01"}


def test_a_section_can_explain_its_own_emptiness() -> None:
    """An absent heading is not information; "no mesh yet" is."""
    builder = ReportBuilder("case", "solver")
    builder.section("mesh", missing="no mesh has been generated yet")
    result = builder.build()
    assert [item.title for item in result.present] == ["mesh"]
    assert result.present[0].missing == "no mesh has been generated yet"


def test_sections_keep_their_order() -> None:
    builder = ReportBuilder("case", "solver")
    for title in ("solver", "time", "mesh"):
        builder.section(title)
        builder.field("x", 1)
    assert [item.title for item in builder.build().present] == ["solver", "time", "mesh"]


def test_an_empty_report_is_falsy() -> None:
    assert not ReportBuilder("case", "solver").build()
    assert not CaseReport(title="x", solver="y")


def test_warnings_are_collected() -> None:
    builder = ReportBuilder("case", "solver")
    builder.warn("something odd")
    assert builder.build().warnings == ("something odd",)


# -- what the adapter extracts --------------------------------------------------------------


def test_the_solver_and_its_settings_are_reported(tmp_path: Path) -> None:
    result = report(case(tmp_path))
    assert fields_of(result, "solver")["Application"] == "pimpleFoam"
    assert fields_of(result, "time")["End time"] == "0.054"
    assert fields_of(result, "time")["Time step"] == "5.4e-07"
    assert fields_of(result, "time")["Start from"] == "startTime"


def test_write_control_is_reported(tmp_path: Path) -> None:
    output = fields_of(report(case(tmp_path)), "output")
    assert output["Write control"] == "timeStep"
    assert output["Write interval"] == "200"
    assert output["Format"] == "binary"


def test_mesh_counts_come_from_the_header_note(tmp_path: Path) -> None:
    """Far cheaper than counting, and it works for a decomposed case too."""
    mesh = fields_of(report(case(tmp_path)), "mesh")
    assert mesh["Cells"] == "383,660"
    assert mesh["Points"] == "770,104"
    assert mesh["Internal faces"] == "765,928"


def test_a_case_with_no_mesh_says_so(tmp_path: Path) -> None:
    result = report(case(tmp_path, owner=None, boundary=None))
    section = next(item for item in result.present if item.title == "mesh")
    assert "no mesh" in section.missing


def test_boundary_patches_are_listed_in_file_order(tmp_path: Path) -> None:
    patches = fields_of(report(case(tmp_path)), "boundary patches")
    assert list(patches) == ["INLET", "TOP", "OUTLET", "WING", "frontAndBackPlanes"]
    assert patches["WING"] == "wall"
    assert patches["frontAndBackPlanes"] == "empty"


def test_fields_and_their_dimensions_are_reported(tmp_path: Path) -> None:
    """Dimensions distinguish a pressure from a kinematic pressure."""
    fields = fields_of(report(case(tmp_path)), "fields")
    assert fields["Present"] == "U, p"
    assert fields["p"] == "[0 2 -2 0 0 0 0]"


def test_the_latest_written_time_is_reported(tmp_path: Path) -> None:
    result = report(case(tmp_path, times=("0.002", "0.004")))
    assert fields_of(result, "time")["Latest written"] == "0.004"
    assert fields_of(result, "output")["Times written"] == "2"


def test_a_case_that_has_written_nothing_says_so(tmp_path: Path) -> None:
    assert "nothing beyond" in fields_of(report(case(tmp_path)), "time")["Latest written"]


def test_decomposition_is_reported(tmp_path: Path) -> None:
    root = case(tmp_path, processors=4)
    (root / "system" / "decomposeParDict").write_text(
        "numberOfSubdomains 4;\nmethod scotch;\n"
    )
    parallel = fields_of(report(root), "parallel")
    assert parallel["Decomposed into"] == "4 subdomains"
    assert parallel["Method"] == "scotch"


def test_a_mismatched_decomposition_is_warned_about(tmp_path: Path) -> None:
    """Submitting will re-decompose, which is worth knowing before it happens."""
    root = case(tmp_path, processors=4)
    (root / "system" / "decomposeParDict").write_text(
        "numberOfSubdomains 16;\nmethod scotch;\n"
    )
    assert any("re-decompose" in warning for warning in report(root).warnings)


def test_a_case_needing_setup_is_warned_about(tmp_path: Path) -> None:
    root = case(tmp_path, fields=())
    (root / "0.orig").mkdir()
    assert any("not been set up" in warning for warning in report(root).warnings)


def test_force_coefficients_are_summarised(tmp_path: Path) -> None:
    root = case(tmp_path)
    path = root / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    path.parent.mkdir(parents=True)
    path.write_text("# Time\tCd\tCl\n1\t0.04\t0.40\n2\t0.05\t0.45\n")

    post = fields_of(report(root), "post-processing")
    assert post["Force coefficients"] == "2 writes"
    assert post["Cl (lift)"] == "0.45"
    assert post["Cd (drag)"] == "0.05"


def test_function_objects_are_listed(tmp_path: Path) -> None:
    root = case(tmp_path, control=CONTROL + "functions { forceCoeffs { } residuals { } }\n")
    assert "forceCoeffs" in fields_of(report(root), "post-processing")["Function objects"]


def test_a_directory_that_is_barely_a_case_still_describes(tmp_path: Path) -> None:
    """The commonest reason to open this page is that the case is incomplete."""
    root = tmp_path / "bare"
    (root / "system").mkdir(parents=True)
    (root / "system" / "controlDict").write_text("application icoFoam;\n")
    result = OpenFOAMAdapter({}).describe_case(CaseContext(workdir=root, cores=1))
    assert fields_of(result, "solver")["Application"] == "icoFoam"


def test_an_adapter_with_no_description_still_produces_a_page(tmp_path: Path) -> None:
    """The default, so no existing adapter had to change and none is required to."""

    class Bare(BaseAdapter):
        name = "bare"

        @classmethod
        def detect(cls, path: Path):
            return None

        def validate(self, ctx):
            raise NotImplementedError

        def plan(self, ctx):
            raise NotImplementedError

    result = Bare({}).describe_case(CaseContext(workdir=tmp_path / "x", cores=1))
    assert result.solver == "bare"
    assert not result


# -- dimensionality and bounds ----------------------------------------------------------------


def test_a_planar_case_is_detected_from_its_empty_patch(tmp_path: Path) -> None:
    """OpenFOAM has no 2-D meshes: an ``empty`` patch is how the user declares one."""
    shape = read_geometry(case(tmp_path))
    assert shape.dimensionality is Dimensionality.TWO_D
    assert shape.is_planar
    assert shape.source is not None and "frontAndBackPlanes" in shape.source


def test_a_volumetric_case_is_detected(tmp_path: Path) -> None:
    shape = read_geometry(case(tmp_path, boundary=BOUNDARY_3D))
    assert shape.dimensionality is Dimensionality.THREE_D
    assert not shape.is_planar


def test_a_case_with_no_boundary_file_is_assumed_volumetric(tmp_path: Path) -> None:
    """"I cannot tell" must not be read as "flat"."""
    shape = read_geometry(case(tmp_path, boundary=None))
    assert shape.dimensionality is Dimensionality.THREE_D
    assert shape.source is None


def test_the_adapter_answers_the_dimensionality_question(tmp_path: Path) -> None:
    """Through the interface the screenshot tool uses, not through the module."""
    shape = OpenFOAMAdapter({}).geometry(CaseContext(workdir=case(tmp_path), cores=1))
    assert shape is not None and shape.is_planar


def test_an_adapter_that_cannot_tell_returns_nothing(tmp_path: Path) -> None:
    from dispatch.adapters.su2 import SU2Adapter

    assert SU2Adapter({}).geometry(CaseContext(workdir=tmp_path, cores=1)) is None


def test_bounds_come_from_the_block_mesh_vertices(tmp_path: Path) -> None:
    """Not from polyMesh/points, which runs to hundreds of megabytes."""
    bounds = read_bounds(case(tmp_path))
    assert bounds is not None
    # scale 0.001 applied: 1000 x 500 x 20 mm becomes 1 x 0.5 x 0.02 m.
    assert bounds.size == pytest.approx((1.0, 0.5, 0.02))


def test_the_scale_factor_is_applied(tmp_path: Path) -> None:
    """A mesh in millimetres read as metres is wrong by a thousand, which would put a
    camera nowhere near the geometry."""
    unscaled = BLOCK_MESH.replace("scale   0.001;", "scale   1;")
    bounds = read_bounds(case(tmp_path, block_mesh=unscaled))
    assert bounds is not None and bounds.size == pytest.approx((1000.0, 500.0, 20.0))


def test_block_indices_are_not_mistaken_for_coordinates(tmp_path: Path) -> None:
    """``hex (0 1 2 3 4 5 6 7)`` read as a vertex would bound the box around indices."""
    bounds = read_bounds(case(tmp_path))
    assert bounds is not None and bounds.maximum[0] == pytest.approx(1.0)


def test_a_case_without_a_block_mesh_has_unknown_bounds(tmp_path: Path) -> None:
    """A snappyHexMesh or imported mesh: the caller asks whatever has the mesh open."""
    assert read_bounds(case(tmp_path, block_mesh=None)) is None


def test_a_planar_case_s_normal_is_its_thinnest_axis(tmp_path: Path) -> None:
    """Real 2-D meshes are a few per cent thick, not a thousandth, so no ratio applies."""
    shape = read_geometry(case(tmp_path))
    assert shape.resolved_normal == 2


def test_the_mesh_counts_tolerate_a_missing_note(tmp_path: Path) -> None:
    root = case(tmp_path, owner="FoamFile { object owner; }\n")
    assert not mesh_counts(root)


def test_patches_are_read_from_a_decomposed_case(tmp_path: Path) -> None:
    """A case worth asking about has usually run, and a parallel run's mesh is in processor0."""
    root = case(tmp_path, boundary=None, owner=None, processors=1)
    (root / "processor0" / "constant" / "polyMesh" / "boundary").write_text(BOUNDARY)
    assert next(patch["name"] for patch in boundary_patches(root)) == "INLET"


def test_bounds_geometry_is_pure_arithmetic() -> None:
    """The camera maths depends on these, so they are pinned independently of any case."""
    bounds = Bounds((0.0, 0.0, 0.0), (2.0, 4.0, 0.1))
    assert bounds.size == pytest.approx((2.0, 4.0, 0.1))
    assert bounds.centre == pytest.approx((1.0, 2.0, 0.05))
    assert bounds.largest == pytest.approx(4.0)
    assert bounds.diagonal == pytest.approx((2**2 + 4**2 + 0.1**2) ** 0.5)
    assert bounds.thinnest_axis == 2
    assert not bounds.is_degenerate


def test_a_degenerate_box_has_no_axes_to_choose_between() -> None:
    point = Bounds((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert point.is_degenerate
    assert point.thinnest_axis is None
    assert point.flat_axis is None


def test_a_declared_plane_beats_a_measurement() -> None:
    """A case can declare itself planar before its mesh exists."""
    declared = CaseGeometry(
        bounds=Bounds((0, 0, 0), (1, 1, 1)),
        dimensionality=Dimensionality.TWO_D,
        normal=1,
    )
    assert declared.resolved_normal == 1


def test_an_unknown_extent_still_carries_the_dimensionality() -> None:
    shape = CaseGeometry(dimensionality=Dimensionality.TWO_D)
    assert shape.is_planar
    assert shape.resolved_normal is None


def test_the_report_dataclasses_are_plain_data() -> None:
    """They cross the IPC boundary, so they hold strings rather than anything live."""
    section = InfoSection(title="t", fields=(InfoField(label="a", value="b"),))
    assert section.fields[0].value == "b"
    assert bool(section)
