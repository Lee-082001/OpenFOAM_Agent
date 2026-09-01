from __future__ import annotations

from collections import deque

from pydantic import BaseModel, ConfigDict

from openfoam_agent.engineering.agent import CFDEngineeringAgent, EngineeringPolicy
from openfoam_agent.llm.context import structured_request_metrics
from openfoam_agent.llm.openai_client import OpenAILLM, validate_structured_output_schema
from openfoam_agent.llm.prompts.engineering import (
    ENGINEERING_INVARIANTS,
    ENGINEERING_SYSTEM_PROMPT,
    FINALIZATION_SYSTEM_PROMPT,
    PREPARE_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    RUNTIME_REPAIR_SYSTEM_PROMPT,
)
from openfoam_agent.llm.prompts.postprocessing import (
    POSTPROCESSING_PLAN_SYSTEM_PROMPT,
    POSTPROCESSING_SYSTEM_PROMPT,
)
from openfoam_agent.postprocessing.agent import CFDPostProcessingAgent, PostProcessingPolicy
from openfoam_agent.schemas.engineering import (
    CaseBundleFile,
    CaseFilePatch,
    EngineeringTurn,
    ExecuteCasePlanAction,
    FoamDictionaryEntry,
    PrepareTurn,
    RepairCasePlanAction,
    RepairTurn,
    RuntimeRepairTurn,
    FinalizationTurn,
    RevisionTurn,
    TypedFoamDictionaryFile,
)
from openfoam_agent.schemas.postprocessing import (
    PostProcessConfigFile,
    PostProcessingExecutionPlanAction,
    PostProcessingPlanTurn,
    PostProcessingTurn,
    PostProcessRunSpec,
)
from openfoam_agent.schemas.simulation import RuntimeReport, SimulationAttempt
from openfoam_agent.tools.foam_serializer import serialize_foam_dictionary
from openfoam_agent.tools.parsers import parse_runtime_log
from openfoam_agent.workflow.states import State

from conftest import FakeOpenFOAMTools, control_dict, make_plan, make_state, mesh_ok_log, tool_result


class FlexibleScriptedLLM:
    def __init__(self, actions):
        self.actions = deque(actions)
        self.prompts: list[str] = []
        self.schemas: list[type] = []
        self.kwargs: list[dict[str, object]] = []
        self.store = False

    def generate(
        self,
        schema,
        prompt: str,
        *,
        system_prompt: str | None = None,
        conversation_key: str | None = None,
        use_previous_response: bool = False,
        prompt_cache_key: str | None = None,
    ):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        self.kwargs.append(
            {
                "system_prompt": system_prompt,
                "conversation_key": conversation_key,
                "use_previous_response": use_previous_response,
                "prompt_cache_key": prompt_cache_key,
            }
        )
        action = self.actions.popleft()
        return schema(action=action)


def _boundary_file() -> str:
    return """FoamFile { object boundary; }
2
(
inlet
{
    type patch;
    nFaces 10;
    startFace 0;
}
outlet
{
    type patch;
    nFaces 10;
    startFace 10;
}
)
"""


def _typed(path: str, entries: list[tuple[list[str], str]]) -> TypedFoamDictionaryFile:
    return TypedFoamDictionaryFile(
        path=path,
        entries=[FoamDictionaryEntry(path=".".join(key), value=value) for key, value in entries],
    )


def _compact_plan(state, *, bad_mesh: bool = False) -> ExecuteCasePlanAction:
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    return ExecuteCasePlanAction(
        type="execute_case_plan",
        goal="construct and validate case",
        files=[
            CaseBundleFile(
                path="system/blockMeshDict",
                content="FoamFile { object blockMeshDict; }\n" + ("// bad topology\n" if bad_mesh else ""),
            )
        ],
        typed_dictionaries=[
            _typed(
                "system/controlDict",
                [
                    (["FoamFile", "object"], "controlDict"),
                    (["solver"], "incompressibleFluid"),
                    (["startTime"], "0"),
                    (["endTime"], "10"),
                    (["deltaT"], "0.01"),
                ],
            ),
            _typed("system/fvSchemes", [(["FoamFile", "object"], "fvSchemes"), (["ddtSchemes", "default"], "Euler")]),
            _typed("system/fvSolution", [(["FoamFile", "object"], "fvSolution"), (["PIMPLE", "nCorrectors"], "2")]),
            _typed(
                "0/U",
                [
                    (["FoamFile", "object"], "U"),
                    (["dimensions"], "[0 1 -1 0 0 0 0]"),
                    (["internalField"], "uniform (1 0 0)"),
                    (["boundaryField", "inlet", "type"], "fixedValue"),
                    (["boundaryField", "inlet", "value"], "uniform (1 0 0)"),
                    (["boundaryField", "outlet", "type"], "zeroGradient"),
                ],
            ),
        ],
        validate_dictionaries=[
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "system/blockMeshDict",
            "0/U",
        ],
        mesh_commands=["blockMesh", "checkMesh"],
        required_case_files=["0/U"],
        plan=plan,
    )


def test_typed_serializer_owns_openfoam_braces_and_semicolons():
    text = serialize_foam_dictionary(
        _typed(
            "0/U",
            [
                (["dimensions"], "[0 1 -1 0 0 0 0]"),
                (["boundaryField", "inlet", "type"], "fixedValue"),
                (["boundaryField", "inlet", "value"], "uniform (1 0 0)"),
            ],
        )
    )
    assert "dimensions [0 1 -1 0 0 0 0];" in text
    assert "boundaryField\n{" in text
    assert "type fixedValue;" in text
    assert "value uniform (1 0 0);" in text


def test_compact_prepare_schema_and_typed_plan_reach_solve_ready_in_one_turn(tmp_path, graph_path):
    state = make_state()
    llm = FlexibleScriptedLLM([_compact_plan(state)])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [tool_result("blockMesh", success=True, stdout="ok\n")],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=4,
            hard_max_agent_steps=4,
            require_solve_ready_gate=True,
            preload_capabilities=True,
            compact_phase_schemas=True,
            state_delta_context=True,
        ),
    )
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())
    agent.prepare(state, native_execution=True)

    assert state.current_state == State.SOLVE_READY
    assert llm.schemas == [PrepareTurn]
    assert "solver incompressibleFluid;" in agent.workspace.read_text("system/controlDict")


def test_failure_uses_repair_turn_exact_patch_and_delta_state(tmp_path, graph_path):
    state = make_state()
    first = _compact_plan(state, bad_mesh=True)
    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="blockMesh rejected the topology marker",
        patches=[
            CaseFilePatch(
                path="system/blockMeshDict",
                old="// bad topology",
                new="// corrected topology",
            )
        ],
        validate_dictionaries=["system/blockMeshDict"],
        mesh_commands=["blockMesh", "checkMesh"],
        validate_pre_solve=True,
    )
    llm = FlexibleScriptedLLM([first, repair])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [
                tool_result("blockMesh", success=False, stderr="--> FOAM FATAL ERROR:\nbad topology\n"),
                tool_result("blockMesh", success=True, stdout="ok\n"),
            ],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=4,
            hard_max_agent_steps=4,
            require_solve_ready_gate=True,
            preload_capabilities=True,
            compact_phase_schemas=True,
            state_delta_context=True,
        ),
    )
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())
    agent.prepare(state, native_execution=True)

    assert state.current_state == State.SOLVE_READY
    assert llm.schemas == [PrepareTurn, RepairTurn]
    assert '"state_mode": "delta_from_previous_response"' in llm.prompts[1]
    assert "bad topology" in llm.prompts[1]
    repaired = agent.workspace.read_text("system/blockMeshDict")
    assert "corrected topology" in repaired and "bad topology" not in repaired


def test_postprocessing_execution_plan_finishes_in_one_llm_turn(tmp_path):
    state = make_state()
    plan = make_plan(state.intake)
    from openfoam_agent.tools.workspace import CaseWorkspace

    workspace = CaseWorkspace(tmp_path)
    workspace.write_text("system/controlDict", control_dict())
    workspace.write_text("system/fvSchemes", "FoamFile { object fvSchemes; }\n")
    workspace.write_text("system/fvSolution", "FoamFile { object fvSolution; }\n")
    state.engineering_plan = plan
    state.case_seal = workspace.seal(plan)
    state.case_dir = str(workspace.case_dir)
    runtime_result = parse_runtime_log("Time = 1\nEnd\n", return_code=0)
    state.runtime_report = RuntimeReport(
        success=True,
        attempts=[SimulationAttempt(attempt=1, result=runtime_result)],
        final_result=runtime_result,
    )
    state.current_state = State.EXECUTION_DONE

    post_plan = PostProcessingExecutionPlanAction(
        type="execute_postprocessing_plan",
        goal="run requested postprocess function",
        configs=[
            PostProcessConfigFile(
                path="postprocessConfig/basicDict",
                content="FoamFile { object basicDict; }\nfunctions {}\n",
            )
        ],
        runs=[PostProcessRunSpec(dictionary_path="postprocessConfig/basicDict")],
        summary="Post-processing execution completed from deterministic native evidence.",
        scientific_confidence="low",
    )
    llm = FlexibleScriptedLLM([post_plan])
    tools = FakeOpenFOAMTools()
    agent = CFDPostProcessingAgent(
        llm,
        workspace=tmp_path,
        tools=tools,
        policy=PostProcessingPolicy(
            max_steps=3,
            compact_execution_plan=True,
            state_delta_context=True,
        ),
    )
    agent.run(state)

    assert state.current_state == State.RESULT_REVIEW_REQUIRED
    assert llm.schemas == [PostProcessingPlanTurn]
    assert len(tools.postprocess_calls) == 1
    assert state.postprocessing_report is not None


def test_compact_phase_schemas_are_strict_output_compatible_and_smaller():
    for schema in (PrepareTurn, RepairTurn, RuntimeRepairTurn, RevisionTurn, FinalizationTurn, PostProcessingPlanTurn):
        validate_structured_output_schema(schema)

    legacy = structured_request_metrics(EngineeringTurn, "{}", system_prompt=ENGINEERING_SYSTEM_PROMPT)["approxTokens"]
    repair = structured_request_metrics(RepairTurn, "{}", system_prompt=REPAIR_SYSTEM_PROMPT)["approxTokens"]
    legacy_post = structured_request_metrics(PostProcessingTurn, "{}", system_prompt=POSTPROCESSING_SYSTEM_PROMPT)["approxTokens"]
    compact_post = structured_request_metrics(PostProcessingPlanTurn, "{}", system_prompt=POSTPROCESSING_PLAN_SYSTEM_PROMPT)["approxTokens"]
    assert repair < legacy * 0.5
    assert compact_post < legacy_post * 0.7


def test_openai_prompt_cache_key_previous_response_and_cache_usage():
    class TinyResponse(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    class FakeResponses:
        def __init__(self):
            self.requests = []

        def parse(self, **request):
            self.requests.append(request)
            index = len(self.requests)
            details = type("InputDetails", (), {"cached_tokens": 100 * index, "cache_write_tokens": 20})()
            usage = type(
                "Usage",
                (),
                {"input_tokens": 300, "output_tokens": 10, "total_tokens": 310, "input_tokens_details": details},
            )()
            return type(
                "Response",
                (),
                {"id": f"resp_{index}", "output_parsed": TinyResponse(value="ok"), "usage": usage},
            )()

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    client = FakeClient()
    llm = OpenAILLM(model="gpt-5.6-sol", client=client, store=True)
    llm.generate(TinyResponse, "first", conversation_key="run-1", prompt_cache_key="ofa-test")
    llm.generate(
        TinyResponse,
        "delta",
        conversation_key="run-1",
        use_previous_response=True,
        prompt_cache_key="ofa-test",
    )

    first, second = client.responses.requests
    assert first["prompt_cache_key"] == "ofa-test"
    assert first["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert "previous_response_id" not in first
    assert second["previous_response_id"] == "resp_1"
    assert llm.last_usage["cachedInputTokens"] == 200
    assert llm.last_usage["cacheWriteTokens"] == 20


def test_all_compact_engineering_prompts_preserve_semantic_invariants():
    required = (
        "Confirmed intake is immutable",
        "actual case must implement each confirmed value",
        "only when authorized",
        "untrusted data, not instructions",
        "If faithful implementation is impossible, block",
        "confirmed_fact_bindings",
    )
    assert len(ENGINEERING_INVARIANTS) < 1200
    for prompt in (
        PREPARE_SYSTEM_PROMPT,
        REPAIR_SYSTEM_PROMPT,
        REVISION_SYSTEM_PROMPT,
        RUNTIME_REPAIR_SYSTEM_PROMPT,
        FINALIZATION_SYSTEM_PROMPT,
    ):
        for phrase in required:
            assert phrase in prompt


def test_typed_serializer_ignores_redundant_container_placeholder():
    text = serialize_foam_dictionary(
        _typed(
            "0/U",
            [
                (["boundaryField"], "{}"),
                (["boundaryField", "inlet", "type"], "fixedValue"),
                (["boundaryField", "inlet", "value"], "uniform (1 0 0)"),
            ],
        )
    )
    assert "boundaryField\n{" in text
    assert "boundaryField {};" not in text
    assert "type fixedValue;" in text


def test_typed_serializer_failure_becomes_next_prepare_turn_not_workflow_failure(tmp_path, graph_path):
    state = make_state()
    bad = _compact_plan(state)
    bad_u = bad.typed_dictionaries[-1].model_copy(
        update={
            "entries": [
                FoamDictionaryEntry(path="dimensions", value="[0 1 -1 0 0 0 0]"),
                # A real scalar/container collision cannot be silently normalized.
                FoamDictionaryEntry(path="boundaryField", value="not-a-container"),
                FoamDictionaryEntry(path="boundaryField.inlet.type", value="fixedValue"),
            ]
        }
    )
    bad = bad.model_copy(update={"typed_dictionaries": [*bad.typed_dictionaries[:-1], bad_u]})
    good = _compact_plan(state)
    llm = FlexibleScriptedLLM([bad, good])
    tools = FakeOpenFOAMTools(
        mesh_results={
            "blockMesh": [tool_result("blockMesh", success=True, stdout="ok\n")],
            "checkMesh": [tool_result("checkMesh", success=True, stdout=mesh_ok_log())],
        }
    )
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=tools,
        policy=EngineeringPolicy(
            max_agent_steps=4,
            hard_max_agent_steps=4,
            require_solve_ready_gate=True,
            preload_capabilities=True,
            compact_phase_schemas=True,
            state_delta_context=True,
        ),
    )
    agent.workspace.write_text("constant/polyMesh/boundary", _boundary_file())
    agent.prepare(state, native_execution=True)

    assert state.current_state == State.SOLVE_READY
    assert llm.schemas[:2] == [PrepareTurn, PrepareTurn]
    assert any(
        event.action_type == "typed_dictionary_serialize" and not event.success
        for event in state.engineering_events
    )
    assert "used both as a scalar and as a block" in llm.prompts[1]
