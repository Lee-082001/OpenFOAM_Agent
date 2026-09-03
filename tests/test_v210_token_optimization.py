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
    PREPARE_DECISION_ONLY_SYSTEM_PROMPT,
    CASE_PLAN_RETRY_SYSTEM_PROMPT,
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
    PrepareDecisionOnlyTurn,
    CasePlanRetryTurn,
    CandidateCasePlanRepairAction,
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
    for schema in (PrepareTurn, PrepareDecisionOnlyTurn, RepairTurn, RuntimeRepairTurn, RevisionTurn, FinalizationTurn, PostProcessingPlanTurn):
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
        PREPARE_DECISION_ONLY_SYSTEM_PROMPT,
        CASE_PLAN_RETRY_SYSTEM_PROMPT,
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
    candidate_repair = CandidateCasePlanRepairAction(
        type="repair_candidate_case_plan",
        diagnosis="Replace only the invalid 0/U candidate dictionary.",
        typed_dictionaries=[good.typed_dictionaries[-1]],
    )
    llm = FlexibleScriptedLLM([bad, candidate_repair])
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
    assert llm.schemas[:2] == [PrepareTurn, CasePlanRetryTurn]
    assert any(
        event.action_type == "typed_dictionary_serialize" and not event.success
        for event in state.engineering_events
    )
    assert "used both as a scalar and as a block" in llm.prompts[1]


def test_plan_only_repair_can_fix_solver_metadata_after_successful_case_execution(tmp_path, graph_path):
    state = make_state()
    first = _compact_plan(state)
    bad_plan = first.plan.model_copy(update={"solver": "foamRunNameHere"})
    first = first.model_copy(update={"plan": bad_plan})

    corrected_plan = bad_plan.model_copy(update={"solver": "incompressibleFluid"})
    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="Case artifacts and mesh evidence are valid; correct only the solver metadata.",
        patches=[],
        replacement_files=[],
        typed_dictionaries=[],
        validate_dictionaries=[],
        surface_checks=[],
        mesh_commands=[],
        validate_pre_solve=False,
        retry_solver=False,
        updated_plan=corrected_plan,
    )
    llm = FlexibleScriptedLLM([first, repair])
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
    assert state.engineering_plan is not None
    assert state.engineering_plan.solver == "incompressibleFluid"
    assert llm.schemas == [PrepareTurn, RepairTurn]


def test_repair_case_plan_allows_plan_only_and_protocol_noop_for_controlled_executor_handling():
    state = make_state()
    plan = make_plan(state.intake).model_copy(update={"required_case_files": ["0/U"]})
    repair = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="metadata-only repair",
        validate_pre_solve=False,
        updated_plan=plan,
    )
    assert repair.updated_plan == plan

    noop = RepairCasePlanAction(
        type="repair_case_plan",
        diagnosis="no-op repair",
        validate_pre_solve=False,
    )
    assert noop.updated_plan is None
    assert not noop.patches and not noop.replacement_files and not noop.typed_dictionaries


def test_case_bundle_preflight_retains_candidate_and_accepts_delta_repair_before_first_write(tmp_path, graph_path):
    state = make_state()
    first = _compact_plan(state)
    unsafe_control = first.typed_dictionaries[0].model_copy(
        update={
            "entries": [
                *first.typed_dictionaries[0].entries,
                FoamDictionaryEntry(path="libs", value='("libsampling.so")'),
            ]
        }
    )
    first = first.model_copy(
        update={"typed_dictionaries": [unsafe_control, *first.typed_dictionaries[1:]]}
    )
    corrected = _compact_plan(state)
    candidate_repair = CandidateCasePlanRepairAction(
        type="repair_candidate_case_plan",
        diagnosis="Remove the non-allowlisted library from the retained controlDict candidate.",
        typed_dictionaries=[corrected.typed_dictionaries[0]],
    )
    llm = FlexibleScriptedLLM([first, candidate_repair])
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
    assert llm.schemas[:2] == [PrepareTurn, CasePlanRetryTurn]
    preflight = [e for e in state.engineering_events if e.action_type == "case_bundle_preflight"]
    assert len(preflight) == 1 and not preflight[0].success
    assert "no candidate case files were written" in preflight[0].summary.lower()
    # The rejected first bundle must not have emitted any write event.  All case
    # writes belong to the corrected second complete plan.
    first_step_writes = [
        e for e in state.engineering_events
        if e.step == 1 and e.action_type == "write_case_file"
    ]
    assert first_step_writes == []
    assert "libsampling.so" in llm.prompts[1]
    assert tools.mesh_calls == ["blockMesh", "checkMesh"]


def test_case_plan_retry_schema_allows_only_candidate_delta_or_block():
    validate_structured_output_schema(CasePlanRetryTurn)
    schema_text = str(CasePlanRetryTurn.model_json_schema())
    assert "SearchReferencesAction" not in schema_text
    assert "ExecuteCasePlanAction" not in schema_text
    assert "CandidateCasePlanRepairAction" in schema_text


def test_repeated_case_bundle_authoring_failures_are_bounded_without_partial_case(tmp_path, graph_path):
    state = make_state()
    base = _compact_plan(state)

    def unsafe_plan():
        unsafe_control = base.typed_dictionaries[0].model_copy(
            update={
                "entries": [
                    *base.typed_dictionaries[0].entries,
                    FoamDictionaryEntry(path="libs", value='("libsampling.so")'),
                ]
            }
        )
        return base.model_copy(
            update={"typed_dictionaries": [unsafe_control, *base.typed_dictionaries[1:]]}
        )

    unsafe = unsafe_plan()
    unsafe_control = unsafe.typed_dictionaries[0]
    repeated_bad_repair = CandidateCasePlanRepairAction(
        type="repair_candidate_case_plan",
        diagnosis="Retry candidate controlDict without changing the rejected library.",
        typed_dictionaries=[unsafe_control],
    )
    llm = FlexibleScriptedLLM([unsafe, repeated_bad_repair, repeated_bad_repair])
    agent = CFDEngineeringAgent(
        llm,
        workspace=tmp_path,
        capability_db=graph_path,
        tools=FakeOpenFOAMTools(),
        policy=EngineeringPolicy(
            max_agent_steps=12,
            hard_max_agent_steps=12,
            compact_phase_schemas=True,
            state_delta_context=True,
            max_case_plan_authoring_retries=3,
        ),
    )
    agent.prepare(state, native_execution=True)

    assert state.current_state == State.ENGINEERING_BLOCKED
    assert llm.schemas == [PrepareTurn, CasePlanRetryTurn, CasePlanRetryTurn]
    assert not any(e.action_type == "write_case_file" for e in state.engineering_events)
    assert sum(e.action_type == "case_bundle_preflight" for e in state.engineering_events) == 3


def test_case_plan_retry_schema_is_compact_candidate_delta_contract():
    metrics = structured_request_metrics(
        CasePlanRetryTurn,
        "{}",
        system_prompt=CASE_PLAN_RETRY_SYSTEM_PROMPT,
    )
    assert metrics["schemaChars"] < 5000
    schema_text = str(CasePlanRetryTurn.model_json_schema())
    assert "ExecuteCasePlanAction" not in schema_text
    assert "EngineeringPlan" not in schema_text
    assert "CandidateCasePlanRepairAction" in schema_text


def test_candidate_repair_schema_accepts_true_noop_for_controlled_executor_handling():
    action = CandidateCasePlanRepairAction(
        type="repair_candidate_case_plan",
        diagnosis="no candidate change",
    )
    assert not action.patches and not action.replacement_files and not action.typed_dictionaries
    assert not action.drop_paths
