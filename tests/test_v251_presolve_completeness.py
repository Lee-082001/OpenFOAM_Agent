from pathlib import Path

from conftest import FakeOpenFOAMTools, ScriptedLLM, make_plan, make_state, mesh_ok_log, tool_result, control_dict
from openfoam_agent.engineering import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.schemas.common import ToolResult
from openfoam_agent.schemas.engineering import FinishPreviewAction, RunMeshCommandAction, SearchCapabilitiesAction, WriteCaseFileAction
from openfoam_agent.tools.diagnostics import diagnose_openfoam_failure
from openfoam_agent.verification.presolve import PreSolveCompletenessGate
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.workflow.states import State


BOUNDARY = '''FoamFile {}\n5\n(\ninlet { type patch; }\noutlet { type patch; }\nfarField { type patch; }\ncylinder { type wall; }\nfrontAndBack { type empty; }\n)\n'''


def field(name: str, *, omit: str | None = None) -> str:
    patches = ["inlet", "outlet", "farField", "cylinder", "frontAndBack"]
    entries = "\n".join(f"  {p} {{ type fixedValue; value uniform 0; }}" for p in patches if p != omit)
    return f'''FoamFile {{ object {name}; }}\ndimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{{\n{entries}\n}}\n'''


def _seed_mesh(workspace: CaseWorkspace) -> None:
    path = workspace.case_dir / "constant/polyMesh/boundary"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BOUNDARY, encoding="utf-8")


def test_presolve_detects_missing_required_file_and_patch_coverage(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    _seed_mesh(workspace)
    workspace.write_text("system/controlDict", control_dict())
    workspace.write_text("system/fvSchemes", "FoamFile {}\nddtSchemes {}\n")
    workspace.write_text("system/fvSolution", "FoamFile {}\nsolvers {}\n")
    workspace.write_text("0/U", field("U", omit="farField"))
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U", "0/p"]})
    gate = PreSolveCompletenessGate(FakeOpenFOAMTools(), workspace)
    result = gate.validate(plan)
    assert not result.valid
    assert any("Required solve input is missing: 0/p" in item for item in result.failures)
    assert any("missing patchField entries: ['farField']" in item for item in result.failures)


def test_presolve_passes_complete_declared_case(tmp_path):
    workspace = CaseWorkspace(tmp_path)
    _seed_mesh(workspace)
    workspace.write_text("system/controlDict", control_dict())
    workspace.write_text("system/fvSchemes", "FoamFile {}\nddtSchemes {}\n")
    workspace.write_text("system/fvSolution", "FoamFile {}\nsolvers {}\n")
    workspace.write_text("0/U", field("U"))
    workspace.write_text("0/p", field("p"))
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U", "0/p"]})
    result = PreSolveCompletenessGate(FakeOpenFOAMTools(), workspace).validate(plan)
    assert result.valid, result.failures
    assert "farField" in result.mesh_patches


def test_cli_policy_style_engineering_reaches_solve_ready_only_after_presolve(tmp_path, graph_path):
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U", "0/p"]})
    actions = [
        SearchCapabilitiesAction(type="search_capabilities", query="incompressibleFluid", rationale="Observe provider."),
        WriteCaseFileAction(type="write_case_file", path="system/controlDict", content=control_dict(), rationale="control"),
        WriteCaseFileAction(type="write_case_file", path="system/fvSchemes", content="FoamFile {}\nddtSchemes {}\n", rationale="schemes"),
        WriteCaseFileAction(type="write_case_file", path="system/fvSolution", content="FoamFile {}\nsolvers {}\n", rationale="solution"),
        WriteCaseFileAction(type="write_case_file", path="0/U", content=field("U"), rationale="U"),
        WriteCaseFileAction(type="write_case_file", path="0/p", content=field("p"), rationale="p"),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale="mesh"),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale="seal solve-ready case"),
    ]
    tools = FakeOpenFOAMTools(mesh_results={"checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())]})
    agent = CFDEngineeringAgent(
        ScriptedLLM(actions), workspace=tmp_path, capability_db=graph_path, tools=tools,
        policy=EngineeringPolicy(max_agent_steps=20, require_solve_ready_gate=True),
    )
    # Native checkMesh fake does not create polyMesh/boundary, seed the deterministic mesh artifact.
    _seed_mesh(agent.workspace)
    agent.prepare(state, native_execution=True)
    assert state.current_state == State.SOLVE_READY
    transitions = [item["to"] for item in state.history]
    assert State.MESH_READY.value in transitions
    assert State.PRE_SOLVE_VALIDATION.value in transitions


def test_sigfpe_startup_banner_does_not_mask_later_foam_fatal_error():
    result = ToolResult(
        success=False,
        command=["foamRun"],
        return_code=1,
        stdout="sigFpe : Enabling floating point exception trapping (FOAM_SIGFPE).\n--> FOAM FATAL ERROR:\ncannot find file system/fvSchemes\n",
    )
    diagnostic = diagnose_openfoam_failure(result, command_name="foamRun")
    assert diagnostic.kind == "foam_fatal_error"
    assert "cannot find file" in diagnostic.excerpt
