from __future__ import annotations

from openfoam_agent.engineering.case_delta_graph import compile_case_delta_graph
from openfoam_agent.contracts.regions import validate_design

import re

from openfoam_agent.schemas.engineering import (
    DeleteCaseFileAction,
    EngineeringPlan,
    ExecuteCasePlanAction,
    CandidateCasePlanRepairAction,
    CandidateBlockMeshRepairAction,
    BlockMeshRepairAction,
    RepairCasePlanAction,
    RuntimeCaseRepairAction,
    RepairTurn,
    StrategyRevisionAction,
    FinishPreviewAction,
    RetrySolverAction,
    RunMeshCommandAction,
    RunNativeOpenFOAMAction,
    SurfaceCheckAction,
    ValidateDictionaryAction,
    ValidatePreSolveAction,
    WriteCaseFileAction,
)
from openfoam_agent.tools.foam_serializer import (
    FoamSerializationError,
    serialize_block_mesh,
    serialize_foam_dictionary,
)
from openfoam_agent.tools.workspace import WorkspaceSafetyError
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State
from openfoam_agent.engineering.phases.repair_episode import record_repair
from openfoam_agent.schemas.simulation import RuntimeRepairDecision
from openfoam_agent.engineering.agent import RepairOutcome


# v4.8 case/runtime repair phase controller. Imported lazily by facade wrappers.


def execute_candidate_block_mesh_repair(
    self,
    state: CFDState,
    repair: CandidateBlockMeshRepairAction,
    *,
    llm_step: int,
    progress_phase: str,
    progress_step: int,
    progress_limit: int,
    native_execution: bool,
) -> bool:
    """Replace only retained structured blockMesh state, then retry whole plan.

    This is a dedicated compact repair path so a topological preflight failure does
    not force the model to regenerate the large generic RepairTurn union or use
    whitespace-sensitive text patches.
    """
    candidate = self._pending_candidate_execution
    if candidate is None:
        event = self._event(
            llm_step, repair.type, False, "No retained candidate blockMesh exists to repair."
        )
        state.engineering_events.append(event)
        return False

    data = candidate.model_dump(mode="python")
    data["files"] = [
        item for item in data.get("files", []) if item.get("path") != "system/blockMeshDict"
    ]
    data["typed_dictionaries"] = [
        item
        for item in data.get("typed_dictionaries", [])
        if item.get("path") != "system/blockMeshDict"
    ]
    data["block_mesh"] = repair.block_mesh.model_dump(mode="python")
    try:
        candidate = ExecuteCasePlanAction.model_validate(data)
    except ValueError as exc:
        event = self._event(llm_step, repair.type, False, f"Candidate blockMesh repair rejected: {exc}")
        state.engineering_events.append(event)
        return False

    self._pending_candidate_execution = candidate
    self._pending_candidate_failed_paths = ("system/blockMeshDict",)
    event = self._event(
        llm_step,
        repair.type,
        True,
        "Replaced retained structured blockMesh candidate; re-running topology/bundle preflight.",
    )
    state.engineering_events.append(event)
    self._emit_engineering_event(
        f"{progress_phase}-candidate-block-mesh-repair",
        event,
        step=progress_step,
        limit=progress_limit,
        state=state,
    )
    return self._execute_case_plan(
        state,
        candidate,
        llm_step=llm_step,
        progress_phase=progress_phase,
        progress_step=progress_step,
        progress_limit=progress_limit,
        native_execution=native_execution,
    )


def execute_candidate_case_plan_repair(
    self,
    state: CFDState,
    repair: CandidateCasePlanRepairAction,
    *,
    llm_step: int,
    progress_phase: str,
    progress_step: int,
    progress_limit: int,
    native_execution: bool,
) -> bool:
    """Repair the retained candidate, then rerun transactional whole-plan preflight."""
    if not (
        repair.patches
        or repair.replacement_files
        or repair.typed_dictionaries
        or repair.drop_paths
    ):
        event = self._event(
            llm_step,
            repair.type,
            False,
            "Candidate repair contained no file change; return a real delta or block.",
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-candidate-repair",
            event,
            step=progress_step,
            limit=progress_limit,
            state=state,
        )
        return False
    try:
        candidate = self._apply_candidate_case_plan_repair(repair)
    except (WorkspaceSafetyError, FoamSerializationError, ValueError) as exc:
        event = self._event(llm_step, repair.type, False, str(exc))
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-candidate-repair",
            event,
            step=progress_step,
            limit=progress_limit,
            state=state,
        )
        return False

    self._pending_candidate_execution = candidate
    event = self._event(
        llm_step,
        repair.type,
        True,
        "Applied delta to retained candidate; re-running whole-bundle authoring preflight.",
    )
    state.engineering_events.append(event)
    self._emit_engineering_event(
        f"{progress_phase}-candidate-repair",
        event,
        step=progress_step,
        limit=progress_limit,
        state=state,
    )
    return self._execute_case_plan(
        state,
        candidate,
        llm_step=llm_step,
        progress_phase=progress_phase,
        progress_step=progress_step,
        progress_limit=progress_limit,
        native_execution=native_execution,
    )


def execute_block_mesh_repair(
    self,
    state: CFDState,
    repair: BlockMeshRepairAction,
    *,
    llm_step: int,
    progress_phase: str,
    native_execution: bool,
) -> bool:
    """Apply one semantic blockMesh replacement and deterministic validation pipeline."""
    plan = state.engineering_plan or self._pending_execution_plan
    if plan is None:
        event = self._event(
            llm_step, repair.type, False, "Structured blockMesh repair has no baseline EngineeringPlan."
        )
        state.engineering_events.append(event)
        return False
    try:
        content = serialize_block_mesh(repair.block_mesh)
    except (FoamSerializationError, ValueError) as exc:
        event = self._event(
            llm_step,
            "block_mesh_repair_preflight",
            False,
            f"Structured blockMesh repair rejected before workspace mutation: {exc}",
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-block-mesh-repair", event, state=state
        )
        return False

    actions: list[object] = [
        WriteCaseFileAction(
            type="write_case_file",
            path=repair.block_mesh.path,
            content=content,
            rationale="",
        ),
        ValidateDictionaryAction(
            type="validate_dictionary", path=repair.block_mesh.path, rationale=""
        ),
        RunMeshCommandAction(type="run_mesh_command", command="blockMesh", rationale=""),
        RunMeshCommandAction(type="run_mesh_command", command="checkMesh", rationale=""),
        ValidatePreSolveAction(
            type="validate_pre_solve",
            required_case_files=plan.required_case_files,
            rationale="",
        ),
        FinishPreviewAction(type="finish_preview", plan=plan, rationale=""),
    ]
    sequence_id = f"{progress_phase}:block-mesh-repair:{llm_step:04d}"
    total = len(actions)
    for index, member in enumerate(actions, start=1):
        if self._tool_action_count(state) >= self.policy.max_tool_actions:
            state.transition(
                State.ENGINEERING_BLOCKED,
                f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).",
            )
            return True
        event, terminal = self._dispatch_prepare(
            state, member, step=llm_step, native_execution=native_execution
        )
        event = event.model_copy(
            update={
                "sequence_id": sequence_id,
                "sequence_goal": repair.diagnosis,
                "sequence_index": index,
                "sequence_length": total,
            }
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-block-mesh-repair",
            event,
            step=index,
            limit=total,
            state=state,
        )
        if not event.success:
            return terminal
        if isinstance(member, WriteCaseFileAction):
            self._structured_block_mesh = repair.block_mesh
        if terminal:
            self._pending_execution_plan = None
            return True
    return False


def repair_actions(
    self,
    state: CFDState,
    repair: RepairCasePlanAction,
    *,
    runtime: bool,
) -> tuple[list[object], EngineeringPlan, object]:
    """Compile a controller-owned delta graph for prepare/runtime repair."""
    baseline_plan = (
        state.pending_revision_plan
        if state.revision_decision_complete and state.pending_revision_plan is not None
        else (state.engineering_plan or self._pending_execution_plan)
    )
    if baseline_plan is None:
        raise WorkspaceSafetyError(
            "Delta repair has no baseline EngineeringPlan. Return execute_case_plan instead."
        )
    if repair.plan_patch is not None:
        try:
            plan = repair.plan_patch.apply(baseline_plan)
        except ValueError as exc:
            raise WorkspaceSafetyError(f"EngineeringPlanPatch rejected: {exc}") from exc
    else:
        plan = repair.updated_plan or baseline_plan
    replacements = [(item.path, item.content) for item in repair.replacement_files]
    replacements += [(item.path, serialize_foam_dictionary(item)) for item in repair.typed_dictionaries]
    patches = [(item.path, item.old, item.new) for item in repair.patches]
    graph = compile_case_delta_graph(
        self.workspace, plan, patches=patches, replacements=replacements,
        native_hints=repair.native_pipeline, mesh_commands=repair.mesh_commands,
        validate_dictionaries=repair.validate_dictionaries, surface_checks=repair.surface_checks,
        validate_pre_solve=repair.validate_pre_solve, phase=("runtime_repair" if runtime else "repair"),
    )
    if not graph.valid:
        raise WorkspaceSafetyError("CaseDeltaGraph rejected repair before mutation: " + " | ".join(graph.failures))
    actions: list[object] = [
        WriteCaseFileAction(type="write_case_file", path=path, content=content, rationale="controller-compiled delta")
        for path, content in graph.changed_files.items()
    ]
    for path in graph.surface_paths:
        actions.append(SurfaceCheckAction(type="surface_check", path=path, rationale="controller-compiled delta validation"))
    legacy_mesh_dispatch = {"blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "createPatch", "checkMesh"}
    for invocation in graph.native_pipeline:
        if invocation.command in legacy_mesh_dispatch and not invocation.arguments:
            actions.append(RunMeshCommandAction(type="run_mesh_command", command=invocation.command, rationale="controller-compiled delta consumer"))
        else:
            actions.append(RunNativeOpenFOAMAction(type="run_openfoam_command", invocation=invocation))
    if graph.validate_pre_solve:
        actions.append(ValidatePreSolveAction(type="validate_pre_solve", required_case_files=plan.required_case_files, rationale="controller-compiled delta manifest"))
    # The controller phase owns the terminal action. ``repair.retry_solver`` is
    # an LLM hint carried by the shared repair schema; it must never promote a
    # prepare/pre-solve repair into runtime solver execution. Prepare repair always
    # returns through finish_preview so seal/solve-readiness and explicit approval
    # gates remain authoritative. Runtime repair always terminates with retry_solver.
    if runtime:
        actions.append(RetrySolverAction(type="retry_solver", plan=plan, rationale=""))
    else:
        actions.append(FinishPreviewAction(type="finish_preview", plan=plan, rationale=""))
    return actions, plan, graph


def execute_prepare_repair_plan(
    self,
    state: CFDState,
    repair: RepairCasePlanAction,
    *,
    llm_step: int,
    progress_phase: str,
    native_execution: bool,
) -> bool:
    if not (
        repair.patches
        or repair.replacement_files
        or repair.typed_dictionaries
        or repair.plan_patch is not None
        or repair.updated_plan is not None
    ):
        event = self._event(
            llm_step,
            repair.type,
            False,
            "Repair plan contained no artifact or EngineeringPlan change; return a real delta or block.",
        )
        state.engineering_events.append(event)
        return False
    try:
        actions, _, graph = self._repair_actions(state, repair, runtime=False)
    except (WorkspaceSafetyError, FoamSerializationError) as exc:
        event = self._event(llm_step, repair.type, False, str(exc))
        state.engineering_events.append(event)
        return False
    sequence_id = f"{progress_phase}:repair-plan:{llm_step:04d}"
    total = len(actions)
    remaining = self.policy.max_tool_actions - self._tool_action_count(state)
    if total > remaining:
        state.transition(State.ENGINEERING_BLOCKED, f"Engineering deterministic action budget cannot cover atomic repair graph ({total} needed, {remaining} remaining).")
        return True
    if native_execution:
        commands = [item.command for item in graph.native_pipeline]
        if graph.surface_paths:
            commands.insert(0, "surfaceCheck")
        ready, failures, _ = self._native_toolchain_preflight(commands)
        if not ready:
            event = self._event(
                llm_step, "native_toolchain_preflight", False,
                "Repair delta requires an unavailable native OpenFOAM toolchain; no case mutation was committed and CFD repair was not recursively invoked.",
                "\n".join(f"- {item}" for item in failures),
                validation_status="inconclusive", failure_category="infra",
            )
            state.engineering_events.append(event)
            self._record_unresolved_failure(state, event)
            self._emit_engineering_event(
                f"{progress_phase}-repair-plan", event, step=1, limit=max(total, 1), state=state
            )
            state.transition(State.ENGINEERING_BLOCKED, event.summary)
            return True
    if progress_phase.startswith("revision") and (graph.changed_files or graph.drop_paths):
        self._begin_confirmed_revision_mutation(state)
    committed = self.workspace.commit_text_transaction(graph.changed_files, drop_paths=graph.drop_paths)
    record_repair(state, diagnosis=repair.diagnosis, changed_files=list(graph.changed_files) + list(graph.drop_paths))
    self._precommitted_files.update(committed)
    self._precommitted_drops.update(graph.drop_paths)
    for index, member in enumerate(actions, start=1):
        if self._tool_action_count(state) >= self.policy.max_tool_actions:
            state.transition(
                State.ENGINEERING_BLOCKED,
                f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).",
            )
            return True
        event, terminal = self._dispatch_prepare(
            state, member, step=llm_step, native_execution=native_execution
        )
        event = event.model_copy(
            update={
                "sequence_id": sequence_id,
                "sequence_goal": repair.diagnosis,
                "sequence_index": index,
                "sequence_length": total,
            }
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-repair-plan",
            event,
            step=index,
            limit=total,
            state=state,
        )
        if not event.success:
            return terminal
        if terminal:
            self._pending_execution_plan = None
            return True
    return False


def execute_strategy_revision(
    self,
    state: CFDState,
    revision: StrategyRevisionAction,
    *,
    llm_step: int,
    progress_phase: str,
    native_execution: bool,
) -> bool:
    """Apply a meshing-strategy delta after tool incompatibility/no-progress.

    Python does not choose the replacement strategy. It only applies the Agent's
    explicit delta, invalidates stale mesh evidence and executes the new pipeline.
    """
    precommit = bool(
        state.engineering_plan is None
        and self._pending_execution_plan is None
        and self._draft_design_plan is not None
    )
    baseline_plan = state.engineering_plan or self._pending_execution_plan or self._draft_design_plan
    if baseline_plan is None:
        event = self._event(
            llm_step, revision.type, False,
            "Strategy revision has no baseline EngineeringPlan; provide plan_patch/updated_plan or block.",
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-strategy-revision", event, step=1, limit=1, state=state
        )
        return False
    if revision.plan_patch is not None:
        try:
            plan = revision.plan_patch.apply(baseline_plan)
        except ValueError as exc:
            event = self._event(
                llm_step, revision.type, False,
                f"EngineeringPlanPatch rejected: {exc}",
            )
            state.engineering_events.append(event)
            return False
    else:
        plan = revision.updated_plan or baseline_plan

    if precommit:
        # No case candidate exists yet. Strategy revision may change only the
        # Python-held EngineeringPlan; authoring will rebuild the complete bundle
        # on the next turn through CaseBuildGraph. Any model-supplied file/native
        # hints are intentionally ignored rather than mutating an uncommitted case.
        failures: list[str] = []
        if revision.plan_patch is None and revision.updated_plan is None:
            failures.append("Pre-commit strategy revision requires plan_patch or updated_plan.")
        if plan.digest() == baseline_plan.digest():
            failures.append("Pre-commit strategy revision did not change the EngineeringPlan.")
        if plan.confirmed_intake_sha256 != state.intake_digest:
            failures.append("Pre-commit strategy revision changed confirmed_intake_sha256.")
        failures.extend(validate_design(plan, state.intake))
        # The staged baseline already passed design provenance validation. Re-run
        # provider provenance only if the revision actually changes execution
        # ownership; a mesh-only feasibility replan must not rediscover the same
        # solver evidence before it can retry authoring.
        execution_changed = (
            plan.solver != baseline_plan.solver
            or plan.solver_provider_id != baseline_plan.solver_provider_id
            or plan.execution != baseline_plan.execution
            or plan.openfoam_version != baseline_plan.openfoam_version
            or plan.openfoam_distribution != baseline_plan.openfoam_distribution
        )
        if execution_changed:
            failures.extend(self._validate_observed_provenance(plan, state))
        failures.extend(self._validate_engineering_defaults(plan, state))
        if failures:
            event = self._event(
                llm_step, revision.type, False,
                "Pre-commit strategy revision was rejected; no case mutation occurred.",
                "\n".join(failures),
                failure_signature="authoring_feasibility:strategy_revision_invalid",
                failure_scope="strategy",
                failure_category="case",
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                f"{progress_phase}-strategy-revision", event, step=1, limit=1, state=state
            )
            return False
        ignored = (
            len(revision.patches)
            + len(revision.replacement_files)
            + len(revision.typed_dictionaries)
            + len(revision.drop_paths)
            + len(revision.mesh_commands)
            + len(revision.native_pipeline)
            + len(revision.validate_dictionaries)
            + len(revision.surface_checks)
            + (1 if revision.block_mesh is not None else 0)
        )
        self._draft_design_plan = plan
        self._authoring_task_queue = None
        self._draft_authoring_brief = ""
        event = self._event(
            llm_step, revision.type, True,
            "Pre-commit engineering strategy revised; complete case authoring will retry from the updated frozen draft.",
            (
                f"ignoredPrecommitFileOrNativeHints={ignored}; "
                f"requiredFiles={len(plan.required_case_files)}"
            ),
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-strategy-revision", event, step=1, limit=1, state=state
        )
        return False

    replacements = [(item.path, item.content) for item in revision.replacement_files]
    replacements += [(item.path, serialize_foam_dictionary(item)) for item in revision.typed_dictionaries]
    if revision.block_mesh is not None:
        replacements.append((revision.block_mesh.path, serialize_block_mesh(revision.block_mesh)))
    patches = [(item.path, item.old, item.new) for item in revision.patches]
    graph = compile_case_delta_graph(
        self.workspace, plan, patches=patches, replacements=replacements,
        drop_paths=revision.drop_paths, native_hints=revision.native_pipeline,
        mesh_commands=revision.mesh_commands, validate_dictionaries=revision.validate_dictionaries,
        surface_checks=revision.surface_checks, validate_pre_solve=revision.validate_pre_solve, phase="strategy_revision",
    )
    if not graph.valid:
        event = self._event(
            llm_step, "case_delta_graph", False,
            "Controller rejected mesh-strategy delta before mutation.",
            "\n".join(graph.failures), validation_status="fail", failure_category="case",
        )
        state.engineering_events.append(event)
        return False
    actions: list[object] = []
    for path in graph.drop_paths:
        actions.append(DeleteCaseFileAction(type="delete_case_file", path=path, rationale="controller-compiled strategy delta"))
    for path, content in graph.changed_files.items():
        actions.append(WriteCaseFileAction(type="write_case_file", path=path, content=content, rationale="controller-compiled strategy delta"))
    for path in graph.surface_paths:
        actions.append(SurfaceCheckAction(type="surface_check", path=path, rationale="controller-compiled strategy validation"))
    legacy_mesh_dispatch = {"blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "createPatch", "checkMesh"}
    for invocation in graph.native_pipeline:
        if invocation.command in legacy_mesh_dispatch and not invocation.arguments:
            actions.append(RunMeshCommandAction(type="run_mesh_command", command=invocation.command, rationale="controller-compiled strategy consumer"))
        else:
            actions.append(RunNativeOpenFOAMAction(type="run_openfoam_command", invocation=invocation))
    if graph.validate_pre_solve:
        actions.append(ValidatePreSolveAction(type="validate_pre_solve", required_case_files=plan.required_case_files, rationale="controller-compiled delta manifest"))
    actions.append(FinishPreviewAction(type="finish_preview", plan=plan, rationale=""))

    sequence_id = f"{progress_phase}:strategy-revision:{llm_step:04d}"
    total = len(actions)
    remaining = self.policy.max_tool_actions - self._tool_action_count(state)
    if total > remaining:
        state.transition(State.ENGINEERING_BLOCKED, f"Engineering deterministic action budget cannot cover atomic strategy graph ({total} needed, {remaining} remaining).")
        return True
    if native_execution:
        commands = [item.command for item in graph.native_pipeline]
        if graph.surface_paths:
            commands.insert(0, "surfaceCheck")
        ready, failures, _ = self._native_toolchain_preflight(commands)
        if not ready:
            event = self._event(
                llm_step, "native_toolchain_preflight", False,
                "Mesh-strategy revision requires an unavailable native OpenFOAM toolchain; no strategy delta was committed.",
                "\n".join(f"- {item}" for item in failures),
                validation_status="inconclusive", failure_category="infra",
            )
            state.engineering_events.append(event)
            self._record_unresolved_failure(state, event)
            self._emit_engineering_event(
                f"{progress_phase}-strategy-revision", event, step=1, limit=max(total, 1), state=state
            )
            state.transition(State.ENGINEERING_BLOCKED, event.summary)
            return True
    committed = self.workspace.commit_text_transaction(graph.changed_files, drop_paths=graph.drop_paths)
    self._precommitted_files.update(committed)
    self._precommitted_drops.update(graph.drop_paths)
    for index, member in enumerate(actions, start=1):
        if self._tool_action_count(state) >= self.policy.max_tool_actions:
            state.transition(State.ENGINEERING_BLOCKED, f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).")
            return True
        event, terminal = self._dispatch_prepare(state, member, step=llm_step, native_execution=native_execution)
        event = event.model_copy(update={
            "sequence_id": sequence_id,
            "sequence_goal": revision.diagnosis,
            "sequence_index": index,
            "sequence_length": total,
        })
        state.engineering_events.append(event)
        self._emit_engineering_event(f"{progress_phase}-strategy-revision", event, step=index, limit=total, state=state)
        if not event.success:
            return terminal
        if (
            revision.block_mesh is not None
            and isinstance(member, WriteCaseFileAction)
            and member.path == revision.block_mesh.path
        ):
            self._structured_block_mesh = revision.block_mesh
        if terminal:
            self._pending_execution_plan = None
            return True
    return False


def runtime_repair_actions(
    self,
    state: CFDState,
    repair: RuntimeCaseRepairAction,
) -> tuple[list[object], EngineeringPlan, object]:
    """Compile runtime edits through the same CaseDeltaGraph as prepare repair."""
    plan = state.engineering_plan
    if plan is None:
        raise WorkspaceSafetyError("Runtime repair requires the approved EngineeringPlan.")
    patches: list[tuple[str, str, str]] = []
    for group in repair.file_patches:
        patches.extend((group.path, edit.old, edit.new) for edit in group.edits)
    replacements = [(item.path, item.content) for item in repair.replacement_files]
    replacements += [(item.path, serialize_foam_dictionary(item)) for item in repair.typed_dictionaries]
    graph = compile_case_delta_graph(
        self.workspace, plan, patches=patches, replacements=replacements,
        native_hints=repair.native_pipeline, mesh_commands=repair.mesh_commands,
        validate_dictionaries=repair.validate_dictionaries, surface_checks=repair.surface_checks,
        validate_pre_solve=repair.validate_pre_solve, phase="runtime_repair",
    )
    if not graph.valid:
        raise WorkspaceSafetyError("CaseDeltaGraph rejected runtime repair before mutation: " + " | ".join(graph.failures))
    actions: list[object] = [
        WriteCaseFileAction(type="write_case_file", path=path, content=content, rationale="controller-compiled runtime delta")
        for path, content in graph.changed_files.items()
    ]
    for path in graph.surface_paths:
        actions.append(SurfaceCheckAction(type="surface_check", path=path, rationale="controller-compiled delta validation"))
    legacy_mesh_dispatch = {"blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "createPatch", "checkMesh"}
    for invocation in graph.native_pipeline:
        if invocation.command in legacy_mesh_dispatch and not invocation.arguments:
            actions.append(RunMeshCommandAction(type="run_mesh_command", command=invocation.command, rationale="controller-compiled runtime consumer"))
        else:
            actions.append(RunNativeOpenFOAMAction(type="run_openfoam_command", invocation=invocation))
    if graph.validate_pre_solve:
        actions.append(ValidatePreSolveAction(type="validate_pre_solve", required_case_files=plan.required_case_files, rationale="controller-compiled delta manifest"))
    if repair.retry_solver:
        actions.append(RetrySolverAction(type="retry_solver", plan=plan, rationale=""))
    return actions, plan, graph


def execute_runtime_repair_plan(
    self,
    state: CFDState,
    repair: RuntimeCaseRepairAction | RepairCasePlanAction,
    *,
    approved_solver: str,
    llm_step: int,
    native_execution: bool,
    runtime_event_start: int,
) -> RepairOutcome | None:
    if isinstance(repair, RuntimeCaseRepairAction) and not (
        repair.file_patches or repair.replacement_files or repair.typed_dictionaries
    ):
        reason = (
            "Runtime repair contained no case-file change; automatic solver retry without "
            "a concrete repair is not authorized."
        )
        state.engineering_events.append(self._event(llm_step, repair.type, False, reason))
        return None
    if isinstance(repair, RepairCasePlanAction) and not (
        repair.patches
        or repair.replacement_files
        or repair.typed_dictionaries
        or repair.plan_patch is not None
        or repair.updated_plan is not None
    ):
        reason = "Runtime repair contained no executable delta."
        state.engineering_events.append(self._event(llm_step, repair.type, False, reason))
        return None
    self._mark_evidence_gaps_satisfied("runtime_repair")
    try:
        if isinstance(repair, RuntimeCaseRepairAction):
            actions, plan, graph = self._runtime_repair_actions(state, repair)
        else:
            actions, plan, graph = self._repair_actions(state, repair, runtime=True)
    except FoamSerializationError as exc:
        state.engineering_events.append(
            self._event(llm_step, repair.type, False, f"Runtime repair serialization failed: {exc}")
        )
        return None
    except WorkspaceSafetyError as exc:
        reason = f"Runtime repair blocked by workspace safety: {exc}"
        return RepairOutcome(RuntimeRepairDecision.BLOCKED, reason=reason)
    if state.execution_approval is not None:
        try:
            state.execution_approval.check_plan(plan)
        except ValueError as exc:
            return RepairOutcome(RuntimeRepairDecision.NEEDS_USER_REVIEW, reason=str(exc))
    if plan.solver != approved_solver:
        return RepairOutcome(
            RuntimeRepairDecision.NEEDS_USER_REVIEW,
            reason="Runtime repair attempted to change the user-approved solver.",
        )
    sequence_id = f"runtime-repair:repair-plan:{llm_step:04d}"
    total = len(actions)
    used = len(state.engineering_events) - runtime_event_start
    remaining = self.policy.max_runtime_repair_tool_actions - used
    if total > remaining:
        return RepairOutcome(RuntimeRepairDecision.BLOCKED, reason=f"Runtime repair action budget cannot cover atomic CaseDeltaGraph ({total} needed, {remaining} remaining).")
    if native_execution:
        commands = [item.command for item in graph.native_pipeline]
        if graph.surface_paths:
            commands.insert(0, "surfaceCheck")
        ready, failures, _ = self._native_toolchain_preflight(commands)
        if not ready:
            event = self._event(
                llm_step, "native_toolchain_preflight", False,
                "Runtime repair requires an unavailable native OpenFOAM toolchain; no repair mutation was committed.",
                "\n".join(f"- {item}" for item in failures),
                validation_status="inconclusive", failure_category="infra",
            )
            state.engineering_events.append(event)
            self._record_unresolved_failure(state, event)
            return RepairOutcome(RuntimeRepairDecision.BLOCKED, reason=event.summary + " " + event.output_excerpt)
    committed = self.workspace.commit_text_transaction(graph.changed_files, drop_paths=graph.drop_paths)
    self._precommitted_files.update(committed)
    self._precommitted_drops.update(graph.drop_paths)
    for index, member in enumerate(actions, start=1):
        if len(state.engineering_events) - runtime_event_start >= self.policy.max_runtime_repair_tool_actions:
            return RepairOutcome(
                RuntimeRepairDecision.BLOCKED,
                reason=f"Runtime repair deterministic action budget exhausted ({self.policy.max_runtime_repair_tool_actions}).",
            )
        outcome: RepairOutcome | None = None
        if isinstance(member, RetrySolverAction):
            event, outcome = self._dispatch_retry_solver(
                state,
                member,
                approved_solver=approved_solver,
                step=llm_step,
                native_execution=native_execution,
            )
        else:
            event = self._dispatch_tool_action(
                member,
                step=llm_step,
                native_execution=native_execution,
                phase="runtime_repair",
                state=state,
            )
        event = event.model_copy(
            update={
                "sequence_id": sequence_id,
                "sequence_goal": repair.diagnosis,
                "sequence_index": index,
                "sequence_length": total,
            }
        )
        state.engineering_events.append(event)
        if not event.success:
            if getattr(self, "_repair_review_required", None):
                return RepairOutcome(RuntimeRepairDecision.NEEDS_USER_REVIEW, reason=self._repair_review_required)
            return outcome
        if outcome is not None:
            return outcome
    return None

