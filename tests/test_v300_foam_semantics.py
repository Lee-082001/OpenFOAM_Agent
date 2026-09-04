from __future__ import annotations

from openfoam_agent.verification.foam_semantics import (
    BoundaryFieldInterpreter,
    BoundaryMatchKind,
    PatternState,
    ResolutionStatus,
    parse_boundary_selectors,
    parse_mesh_boundary,
    parse_top_level_assignments,
)
from openfoam_agent.verification.presolve import PreSolveCompletenessGate
from openfoam_agent.tools.workspace import CaseWorkspace

from conftest import FakeOpenFOAMTools, control_dict, foam_header, make_plan, make_state


MESH = r'''FoamFile {}
5
(
    inlet { type patch; }
    outlet { type patch; }
    wall1
    {
        type wall;
        inGroups 2(solidWalls walls);
    }
    obstacle
    {
        type wall;
        inGroups (solidWalls);
    }
    frontAndBack { type empty; }
)
'''


def field(boundary_body: str) -> str:
    return foam_header("0/U", "volVectorField") + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
{boundary_body}
}}
"""


def test_mesh_ir_preserves_patch_type_groups_and_order():
    mesh = parse_mesh_boundary(MESH)
    assert mesh.names == ["inlet", "outlet", "wall1", "obstacle", "frontAndBack"]
    wall1 = mesh.patches[2]
    assert wall1.patch_type == "wall"
    assert wall1.groups == frozenset({"solidWalls", "walls"})
    assert wall1.order == 2


def test_quoted_literal_is_not_automatically_regex():
    selectors = parse_boundary_selectors(field('''    "walls" { type noSlip; }'''))
    assert len(selectors) == 1
    assert selectors[0].key.value == "walls"
    assert selectors[0].key.pattern_state == PatternState.LITERAL


def test_parenthesized_regex_covers_multiple_patches():
    mesh = parse_mesh_boundary(MESH)
    selectors = parse_boundary_selectors(field(r'''    "(wall1|obstacle)" { type noSlip; }
    inlet { type fixedValue; }
    outlet { type zeroGradient; }
'''))
    resolved = BoundaryFieldInterpreter().resolve_all(mesh, selectors)
    assert resolved["wall1"].match_kind == BoundaryMatchKind.REGEX
    assert resolved["obstacle"].match_kind == BoundaryMatchKind.REGEX


def test_exact_overrides_later_regex():
    mesh = parse_mesh_boundary(MESH)
    selectors = parse_boundary_selectors(field(r'''    frontAndBack { type empty; }
    ".*" { type zeroGradient; }
'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["frontAndBack"]
    assert resolution.match_kind == BoundaryMatchKind.EXACT
    assert resolution.effective_field_type == "empty"


def test_group_overrides_later_regex():
    mesh = parse_mesh_boundary(MESH)
    selectors = parse_boundary_selectors(field(r'''    solidWalls { type noSlip; }
    "wall.*" { type slip; }
'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["wall1"]
    assert resolution.match_kind == BoundaryMatchKind.GROUP
    assert resolution.effective_field_type == "noSlip"



def test_exact_overrides_group():
    mesh = parse_mesh_boundary(MESH)
    selectors = parse_boundary_selectors(field(r'''    solidWalls { type noSlip; }
    wall1 { type fixedValue; }'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["wall1"]
    assert resolution.match_kind == BoundaryMatchKind.EXACT
    assert resolution.effective_field_type == "fixedValue"


def test_later_group_wins_within_group_tier():
    mesh = parse_mesh_boundary(MESH)
    selectors = parse_boundary_selectors(field(r'''    solidWalls { type noSlip; }
    walls { type slip; }'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["wall1"]
    assert resolution.match_kind == BoundaryMatchKind.GROUP
    assert resolution.selector is not None
    assert resolution.selector.key.value == "walls"
    assert resolution.effective_field_type == "slip"


def test_later_regex_wins_within_regex_tier():
    mesh = parse_mesh_boundary(MESH)
    selectors = parse_boundary_selectors(field(r'''    "wall.*" { type slip; }
    ".*" { type zeroGradient; }
'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["wall1"]
    assert resolution.match_kind == BoundaryMatchKind.REGEX
    assert resolution.selector is not None
    assert resolution.selector.key.raw == '".*"'
    assert resolution.effective_field_type == "zeroGradient"


def test_auto_empty_precedes_regex_fallback():
    mesh = parse_mesh_boundary(MESH)
    selectors = parse_boundary_selectors(field(r'''    ".*" { type zeroGradient; }'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["frontAndBack"]
    assert resolution.match_kind == BoundaryMatchKind.AUTO_EMPTY
    assert resolution.effective_field_type == "empty"


def test_same_non_pattern_entry_can_match_exact_and_group():
    mesh_text = '''FoamFile {}\n3\n(\nwall { type wall; }\nwall1 { type wall; inGroups (wall); }\nwall2 { type wall; inGroups (wall); }\n)\n'''
    mesh = parse_mesh_boundary(mesh_text)
    selectors = parse_boundary_selectors(field('''    wall { type noSlip; }'''))
    resolved = BoundaryFieldInterpreter().resolve_all(mesh, selectors)
    assert resolved["wall"].match_kind == BoundaryMatchKind.EXACT
    assert resolved["wall1"].match_kind == BoundaryMatchKind.GROUP
    assert resolved["wall2"].match_kind == BoundaryMatchKind.GROUP


def test_dynamic_selector_becomes_indeterminate_not_missing():
    mesh_text = '''FoamFile {}\n1\n(\nobstacle { type wall; }\n)\n'''
    mesh = parse_mesh_boundary(mesh_text)
    selectors = parse_boundary_selectors(field('''    $obstacleBC;'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["obstacle"]
    assert resolution.status == ResolutionStatus.INDETERMINATE
    assert resolution.match_kind == BoundaryMatchKind.INDETERMINATE



def test_dynamic_expansion_keeps_visible_regex_candidate_indeterminate():
    mesh_text = '''FoamFile {}\n1\n(\nobstacle { type wall; }\n)\n'''
    mesh = parse_mesh_boundary(mesh_text)
    selectors = parse_boundary_selectors(field(r'''    $extraBoundaryEntries;
    ".*" { type zeroGradient; }'''))
    resolution = BoundaryFieldInterpreter().resolve_all(mesh, selectors)["obstacle"]
    assert resolution.status == ResolutionStatus.INDETERMINATE


def test_presolve_uses_semantic_resolution_for_coverage_and_constraints(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    boundary = workspace.case_dir / "constant/polyMesh/boundary"
    boundary.parent.mkdir(parents=True, exist_ok=True)
    boundary.write_text(MESH, encoding="utf-8")
    workspace.write_text("system/controlDict", control_dict())
    workspace.write_text("system/fvSchemes", foam_header("system/fvSchemes") + "ddtSchemes {}\n")
    workspace.write_text("system/fvSolution", foam_header("system/fvSolution") + "solvers {}\n")
    workspace.write_text(
        "0/U",
        field(r'''    inlet { type fixedValue; }
    outlet { type zeroGradient; }
    solidWalls { type noSlip; }
    ".*" { type zeroGradient; }
'''),
    )
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    result = PreSolveCompletenessGate(FakeOpenFOAMTools(), workspace).validate(plan)
    assert result.valid, result.failures
    assert result.boundary_resolutions["0/U"]["wall1"] == "group"
    assert result.boundary_resolutions["0/U"]["frontAndBack"] == "auto_empty"


def test_presolve_reports_indeterminate_as_warning_not_false_missing(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    boundary = workspace.case_dir / "constant/polyMesh/boundary"
    boundary.parent.mkdir(parents=True, exist_ok=True)
    boundary.write_text("FoamFile {}\n1\n(\nobstacle { type wall; }\n)\n", encoding="utf-8")
    workspace.write_text("system/controlDict", control_dict())
    workspace.write_text("system/fvSchemes", foam_header("system/fvSchemes") + "ddtSchemes {}\n")
    workspace.write_text("system/fvSolution", foam_header("system/fvSolution") + "solvers {}\n")
    workspace.write_text("0/U", field("    $obstacleBC;"))
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    result = PreSolveCompletenessGate(FakeOpenFOAMTools(), workspace).validate(plan)
    assert result.valid, result.failures
    assert result.warnings
    assert "did not prove this patch missing" in result.warnings[0]


def test_top_level_field_projection_ignores_comment_substrings():
    text = foam_header("0/U", "volVectorField") + """
// dimensions [0 1 -1 0 0 0 0];
// internalField uniform (0 0 0);
boundaryField { inlet { type zeroGradient; } }
"""
    entries, complete = parse_top_level_assignments(text)
    assert complete
    assert "dimensions" not in entries
    assert "internalField" not in entries
    assert entries["boundaryField"] == "<dictionary>"


def test_top_level_field_projection_marks_dynamic_include_indeterminate_without_swallowing_later_keys():
    text = foam_header("0/U", "volVectorField") + """
#include "fieldDefaults"
internalField uniform (0 0 0);
boundaryField { inlet { type zeroGradient; } }
"""
    entries, complete = parse_top_level_assignments(text)
    assert not complete
    assert entries["internalField"] == "uniform (0 0 0)"
    assert "dimensions" not in entries
