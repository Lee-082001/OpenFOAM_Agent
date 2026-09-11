from __future__ import annotations

from openfoam_agent.contracts.evidence import implementation_evidence_pack, evidence_coverage_failures, authoring_prompt_evidence, advisory_authoring_evidence_summary
from openfoam_agent.contracts.evidence_policy import POLICY_SUMMARY, provider_is_sufficient
from openfoam_agent.engineering.authoring_tasks import compile_tasks, accept_task
from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.engineering.case_delta_graph import compile_case_delta_graph
from openfoam_agent.llm.context import ContextBudgetError
from openfoam_agent.llm.context_capsules import project_confirmed_intake, project_plan_core
from openfoam_agent.engineering.design_context import build_partitioned_design_prompt
from openfoam_agent.engineering.revision_context import (
    build_partitioned_revision_prompt,
    project_revision_plan,
    project_revision_proposal,
    project_strategy_plan,
)
from openfoam_agent.engineering.repair_context import (
    build_partitioned_validation_repair_prompt,
    project_validation_repair_plan,
)
from openfoam_agent.contracts.regions import region_layouts, region_mesh_digest, validate_design
from openfoam_agent.tools.execution_policy import (
    deny_unapproved_engineering_solve, command_effect, ExecutionPolicyError,
)
from openfoam_agent.tools.safe_runner import SafeRunner

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openfoam_agent.agents.intake import confirmed_intake_definition
from openfoam_agent.llm.context import (
    build_bounded_json_prompt,
    compact_event_for_model,
    compact_runtime_report,
    compact_text,
    structured_request_metrics,
)
from openfoam_agent.llm.prompts import (
    ENGINEERING_SYSTEM_PROMPT,
    PREPARE_SYSTEM_PROMPT,
    PREPARE_DECISION_ONLY_SYSTEM_PROMPT,
    PREPARE_DESIGN_SYSTEM_PROMPT,
    PREPARE_DECISION_DESIGN_SYSTEM_PROMPT,
    CASE_AUTHORING_SYSTEM_PROMPT,
    CASE_PLAN_RETRY_SYSTEM_PROMPT,
    CANDIDATE_BLOCK_MESH_REPAIR_SYSTEM_PROMPT,
    BLOCK_MESH_REPAIR_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    REVISION_DECISION_SYSTEM_PROMPT,
    REVISION_AUTHORING_SYSTEM_PROMPT,
    RUNTIME_REPAIR_SYSTEM_PROMPT,
    STRATEGY_REVISION_SYSTEM_PROMPT,
    FINALIZATION_SYSTEM_PROMPT,
)
from openfoam_agent.llm.protocol import StructuredLLM
from openfoam_agent.progress import (
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    action_importance,
    describe_action,
)
from openfoam_agent.schemas.feedback import RevisionFileChange, RevisionRecord
from openfoam_agent.schemas.engineering import (
    BlockAction,
    CaseBundleFile,
    DeleteCaseFileAction,
    EngineeringBudgetExtension,
    EngineeringEvent,
    ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT,
    EngineeringEvidenceRecord,
    EngineeringPlan,
    EngineeringSequenceAction,
    ExecuteCasePlanAction,
    DesignCaseAction,
    CaseAuthoringAction,
    PrepareDesignTurn,
    PrepareDecisionDesignTurn,
    CaseAuthoringTurn,
    FinalizationTurn,
    PrepareTurn,
    PrepareDecisionOnlyTurn,
    CasePlanRetryTurn,
    CandidateCasePlanRepairAction,
    CandidateBlockMeshRepairAction,
    CandidateBlockMeshRepairTurn,
    BlockMeshRepairAction,
    BlockMeshRepairTurn,
    GatherEvidenceAction,
    EvidenceGapRequest,
    RepairCasePlanAction,
    RuntimeCaseRepairAction,
    RepairTurn,
    RevisionTurn,
    RevisionDecisionAction,
    RevisionDecisionTurn,
    RevisionAuthoringTurn,
    RuntimeRepairTurn,
    StrategyRevisionAction,
    StrategyRevisionTurn,
    TypedBlockMeshFile,
    EngineeringTurn,
    ObservedEngineeringEvidence,
    canonical_engineering_evidence_id,
    FinishPreviewAction,
    InspectEnvironmentAction,
    ListCaseFilesAction,
    ReadCaseFileAction,
    ReadReferenceAction,
    PatchCaseFileAction,
    RetrySolverAction,
    RunMeshCommandAction,
    RunNativeOpenFOAMAction,
    SearchCapabilitiesAction,
    SearchReferencesAction,
    SurfaceCheckAction,
    ValidateDictionaryAction,
    ValidatePreSolveAction,
    WriteCaseFileAction,
)
from openfoam_agent.tools.capability_catalog import CapabilityCatalog
from openfoam_agent.tools.diagnostics import diagnose_openfoam_failure, classify_native_validation
from openfoam_agent.tools.foam_file import validate_foam_file_header
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.foam_serializer import (
    FoamSerializationError,
    serialize_block_mesh,
    serialize_foam_dictionary,
)
from openfoam_agent.tools.references import OpenFOAMReferenceIndex
from openfoam_agent.tools.workspace import CaseWorkspace, WorkspaceSafetyError
from openfoam_agent.verification.presolve import PreSolveCompletenessGate
from openfoam_agent.verification.safety import (
    DeterministicSafetyGate,
    parse_check_mesh_evidence,
)
from openfoam_agent.workflow.state import CFDState
from openfoam_agent.workflow.states import State
from openfoam_agent.engineering.phases.repair_episode import ensure_episode, add_support_observation, record_repair
from openfoam_agent.schemas.simulation import RuntimeRepairDecision

# v4.8 initial/candidate authoring phase controller.


def candidate_failure_paths(
    failures: list[str],
    candidate_bundle: dict[str, str],
) -> list[str]:
    """Extract implicated authored paths from deterministic preflight diagnostics."""
    matched: list[str] = []
    for path in candidate_bundle:
        if any(path in failure for failure in failures):
            matched.append(path)
    return matched[:12]


def candidate_repair_context(self) -> dict[str, object] | None:
    """Return a bounded capsule for repairing the retained in-memory candidate."""
    candidate = self._pending_candidate_execution
    if candidate is None:
        return None
    failed = set(self._pending_candidate_failed_paths)
    manifest: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for item in candidate.files:
        manifest.append({"path": item.path, "kind": "raw", "chars": len(item.content)})
        if item.path in failed:
            artifacts.append(
                {"path": item.path, "kind": "raw", "content": item.content[:40_000]}
            )
    for item in candidate.typed_dictionaries:
        manifest.append(
            {
                "path": item.path,
                "kind": "typed_dictionary",
                "foam_class": item.foam_class,
                "entries": len(item.entries),
            }
        )
        if item.path in failed:
            artifacts.append(
                {
                    "path": item.path,
                    "kind": "typed_dictionary",
                    "foam_class": item.foam_class,
                    "entries": [entry.model_dump(mode="json") for entry in item.entries],
                }
            )
    if candidate.block_mesh is not None:
        manifest.append(
            {
                "path": candidate.block_mesh.path,
                "kind": "block_mesh",
                "vertices": len(candidate.block_mesh.vertices),
                "blocks": len(candidate.block_mesh.blocks),
                "boundary_patches": len(candidate.block_mesh.boundary),
            }
        )
        if candidate.block_mesh.path in failed:
            artifacts.append(
                {
                    "path": candidate.block_mesh.path,
                    "kind": "block_mesh",
                    "spec": candidate.block_mesh.model_dump(mode="json"),
                }
            )
    authored_paths = {item["path"] for item in manifest}
    missing_required = [
        path for path in candidate.plan.required_case_files if path not in authored_paths
    ]
    return {
        "goal": candidate.goal,
        "failed_paths": list(self._pending_candidate_failed_paths),
        "missing_required_files": missing_required,
        "manifest": manifest,
        "failed_artifacts": artifacts,
        "controller_build_policy": {
            "manifest_authority": "EngineeringPlan.required_case_files",
            "instruction": (
                "Repair only missing/conflicting authored artifacts. Do not redesign validation lists; "
                "Python compiles static validation, surface checks, mesh consumers and final checkMesh "
                "from the completed bundle."
            ),
        },
        "pipeline": {
            "validate_dictionaries": candidate.validate_dictionaries,
            "surface_checks": candidate.surface_checks,
            "mesh_commands": candidate.mesh_commands,
            "required_case_files": candidate.required_case_files,
        },
        "plan_capsule": {
            "solver": candidate.plan.solver,
            "solver_provider_id": candidate.plan.solver_provider_id,
            "required_case_files": candidate.plan.required_case_files,
            "confirmed_intake_sha256": candidate.plan.confirmed_intake_sha256,
        },
    }


def apply_candidate_case_plan_repair(
    self,
    repair: CandidateCasePlanRepairAction,
) -> ExecuteCasePlanAction:
    """Apply a model-authored delta to the retained candidate without touching workspace."""
    candidate = self._pending_candidate_execution
    if candidate is None:
        raise WorkspaceSafetyError("No retained candidate case plan exists to repair.")

    raw = {item.path: item for item in candidate.files}
    typed = {item.path: item for item in candidate.typed_dictionaries}
    block_mesh = candidate.block_mesh

    for path in repair.drop_paths:
        raw.pop(path, None)
        typed.pop(path, None)
        if block_mesh is not None and block_mesh.path == path:
            block_mesh = None

    for patch in repair.patches:
        if patch.path in raw:
            current = raw[patch.path].content
        elif patch.path in typed:
            current = serialize_foam_dictionary(typed[patch.path])
        else:
            raise WorkspaceSafetyError(
                f"Candidate patch target does not exist: {patch.path}"
            )
        if current.count(patch.old) != 1:
            raise WorkspaceSafetyError(
                f"Candidate patch old fragment must occur exactly once in {patch.path}."
            )
        content = current.replace(patch.old, patch.new, 1)
        raw[patch.path] = CaseBundleFile(path=patch.path, content=content)
        typed.pop(patch.path, None)

    replacement_seen: dict[str, str] = {}
    for item in repair.replacement_files:
        prior = replacement_seen.get(item.path)
        if prior is not None and prior != item.content:
            raise WorkspaceSafetyError(f"Conflicting candidate replacement content for {item.path}.")
        replacement_seen[item.path] = item.content
        if item.path in typed:
            raise WorkspaceSafetyError(f"Candidate delta cannot replace {item.path} in raw and typed form in one turn.")
        raw[item.path] = item
        if block_mesh is not None and block_mesh.path == item.path:
            block_mesh = None

    typed_seen: dict[str, str] = {}
    for item in repair.typed_dictionaries:
        rendered = serialize_foam_dictionary(item)
        prior = typed_seen.get(item.path)
        if prior is not None and prior != rendered:
            raise WorkspaceSafetyError(f"Conflicting candidate typed content for {item.path}.")
        typed_seen[item.path] = rendered
        if item.path in raw and item.path not in replacement_seen:
            raw.pop(item.path, None)
        elif item.path in replacement_seen:
            raise WorkspaceSafetyError(f"Candidate delta cannot replace {item.path} in raw and typed form in one turn.")
        typed[item.path] = item
        if block_mesh is not None and block_mesh.path == item.path:
            block_mesh = None

    data = candidate.model_dump(mode="python")
    data["files"] = [item.model_dump(mode="python") for item in raw.values()]
    data["typed_dictionaries"] = [item.model_dump(mode="python") for item in typed.values()]
    data["block_mesh"] = block_mesh.model_dump(mode="python") if block_mesh is not None else None
    return ExecuteCasePlanAction.model_validate(data)


def execute_case_plan(
    self,
    state: CFDState,
    execution: ExecuteCasePlanAction,
    *,
    llm_step: int,
    progress_phase: str,
    progress_step: int,
    progress_limit: int,
    native_execution: bool,
) -> bool:
    """Execute a complete LLM-authored case plan without intermediate LLM calls.

    The high-level plan is deliberately expanded into the existing primitive
    actions. This preserves the exact same sandbox, path checks, OpenFOAM
    command allowlists, budgets, checkMesh evidence parser, pre-solve gate and
    final plan/CaseSeal validation used by ordinary actions.
    """

    # v4.5.1: structured-output/Pydantic accepts harmless duplicate authoring
    # echoes and carries substantive conflicts to this deterministic boundary.
    # Do not spend a full Codex structured-output retry on a conflict that can be
    # repaired as a compact retained-candidate delta.
    authoring_conflicts = list(execution.authoring_conflicts)
    conflict_paths: list[str] = []
    for item in execution.typed_dictionaries:
        if item.entry_conflicts:
            conflict_paths.append(item.path)
            authoring_conflicts.extend(
                f"typed-entry:{item.path}:{entry_path}" for entry_path in item.entry_conflicts
            )
    if authoring_conflicts:
        for token in authoring_conflicts:
            parts = str(token).split(":")
            for candidate in parts:
                if re.fullmatch(r"(?:0|constant|system|postprocessConfig)/[A-Za-z0-9_.\/-]+", candidate):
                    conflict_paths.append(candidate)
        event = self._event(
            llm_step,
            "authoring_semantic_conflict",
            False,
            "Case authoring contains conflicting duplicate representations; retained candidate repair is required.",
            "\n".join(f"- {item}" for item in authoring_conflicts[:40]),
            validation_status="fail",
            failure_category="case",
        )
        blocked = self._record_case_plan_authoring_failure(state, event)
        self._emit_engineering_event(
            f"{progress_phase}-execution-plan",
            event,
            step=progress_step,
            limit=progress_limit,
            state=state,
        )
        self._pending_candidate_execution = execution
        self._pending_candidate_failed_paths = tuple(dict.fromkeys(conflict_paths))[:12]
        self._pending_execution_plan = None
        return blocked

    actions: list[object] = []
    rendered_files: list[tuple[str, str]] = [
        (item.path, item.content) for item in execution.files
    ]
    for item in execution.typed_dictionaries:
        try:
            content = serialize_foam_dictionary(item)
        except FoamSerializationError as exc:
            event = self._event(
                llm_step,
                "typed_dictionary_serialize",
                False,
                f"Typed dictionary serialization failed for {item.path}: {exc}",
            )
            blocked = self._record_case_plan_authoring_failure(state, event)
            self._emit_engineering_event(
                f"{progress_phase}-execution-plan",
                event,
                step=progress_step,
                limit=progress_limit,
                state=state,
            )
            # Authoring failures are transactional: no candidate case file has
            # been committed. Retain the complete candidate in memory so the next
            # turn can repair only the implicated candidate entry instead of
            # regenerating the whole case as a large Structured Output object.
            self._pending_candidate_execution = execution
            self._pending_candidate_failed_paths = (item.path,)
            self._pending_execution_plan = None
            return blocked
        rendered_files.append((item.path, content))

    if execution.block_mesh is not None:
        try:
            rendered_files.append((execution.block_mesh.path, serialize_block_mesh(execution.block_mesh)))
            # Preserve the semantically validated structured candidate even when a
            # later manifest-coverage check finds another missing file. This keeps
            # blockMesh repair state available without claiming the case was committed.
            self._structured_block_mesh = execution.block_mesh
        except (FoamSerializationError, ValueError) as exc:
            event = self._event(
                llm_step,
                "block_mesh_serialize",
                False,
                f"Typed blockMesh serialization failed: {exc}",
            )
            blocked = self._record_case_plan_authoring_failure(state, event)
            self._emit_engineering_event(
                f"{progress_phase}-execution-plan", event, step=progress_step, limit=progress_limit, state=state
            )
            self._pending_candidate_execution = execution
            self._pending_candidate_failed_paths = ("system/blockMeshDict",)
            self._pending_execution_plan = None
            return blocked

    # v2.11: all-or-nothing authoring preflight.  Validate path/content/library
    # policy and aggregate authored size for the *entire* candidate bundle before
    # the first workspace mutation.  This prevents an early rejected file from
    # leaving only the preceding files behind and triggering a long missing-file
    # repair cascade.
    candidate_bundle = {path: content for path, content in rendered_files}

    # A later repair turn may intentionally provide only changed files after an
    # earlier complete candidate was committed and a native consumer failed. Reuse
    # only controller-tracked authored files from this workspace as the immutable
    # baseline; an initial incomplete candidate has no such baseline and therefore
    # cannot pass manifest coverage accidentally.
    effective_bundle = dict(candidate_bundle)
    existing_authored = set(self.workspace.list_authored())
    reused_existing_paths: list[str] = []
    for required_path in execution.plan.required_case_files:
        if required_path in effective_bundle or required_path not in existing_authored:
            continue
        existing_path = self.workspace.resolve_case_path(required_path, must_exist=True)
        effective_bundle[required_path] = existing_path.read_text(encoding="utf-8", errors="replace")
        reused_existing_paths.append(required_path)

    # v4.6: compile one controller-owned build graph before the first workspace
    # mutation. The frozen EngineeringPlan is the manifest authority; model-authored
    # validation lists can never create actions for files that were not authored.
    build_graph = compile_case_build_graph(execution, effective_bundle)
    if not build_graph.valid:
        event = self._event(
            llm_step,
            "case_build_graph",
            False,
            "Controller could not compile a complete case build graph; no candidate files were written.",
            "\n".join(f"- {failure}" for failure in build_graph.failures),
            validation_status="fail",
            failure_category="case",
        )
        blocked = self._record_case_plan_authoring_failure(state, event)
        self._emit_engineering_event(
            f"{progress_phase}-execution-plan",
            event,
            step=progress_step,
            limit=progress_limit,
            state=state,
        )
        self._pending_candidate_execution = execution
        self._pending_candidate_failed_paths = tuple(build_graph.missing_required_paths)[:20]
        self._pending_execution_plan = None
        return blocked

    bundle_failures = self.workspace.validate_candidate_bundle(candidate_bundle)
    # v4.2.1 progress-first authoring: syntax/reference evidence is advisory
    # provenance, not a write permission token. Raw and typed files are both
    # admitted to the deterministic safety/header/parser/native pipeline below.
    # Security-sensitive directives, unsafe paths/libraries and native commands
    # remain fail-closed in CaseWorkspace/SafeRunner.

    # v3.0.2: solve-critical OpenFOAM files must satisfy the IOobject-facing
    # FoamFile contract before *any* candidate file is committed. This closes the
    # gap where a shallow dictionary probe accepted headerless content and blockMesh/foamRun
    # discovered the malformed header later, one file at a time.
    # Static/header targets are compiled from the artifacts that actually exist,
    # not from a redundant model-authored validation mirror. Surface/data inputs
    # are intentionally left to their real consumers.
    header_targets = list(build_graph.dictionary_paths)
    for path in header_targets:
        content = effective_bundle.get(path)
        if content is None:
            continue
        suffix = Path(path).suffix.lower()
        if suffix in {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".emesh"}:
            continue
        header = validate_foam_file_header(
            path,
            content,
            expected_class=("dictionary" if path.startswith("system/") else None),
        )
        bundle_failures.extend(
            f"{path}: {failure}" for failure in header.failures
        )

    if bundle_failures:
        event = self._event(
            llm_step,
            "case_bundle_preflight",
            False,
            "Case bundle rejected before commit; no candidate case files were written.",
            "\n".join(f"- {failure}" for failure in bundle_failures),
        )
        blocked = self._record_case_plan_authoring_failure(state, event)
        self._emit_engineering_event(
            f"{progress_phase}-execution-plan",
            event,
            step=progress_step,
            limit=progress_limit,
            state=state,
        )
        self._pending_candidate_execution = execution
        self._pending_candidate_failed_paths = tuple(
            self._candidate_failure_paths(bundle_failures, candidate_bundle)
        )
        self._pending_execution_plan = None
        return blocked

    # v4.7.2: selected native utilities must be loadable under the exact
    # sanitized OpenFOAM environment before the first candidate write. This is
    # an infrastructure gate, not a CFD repair trigger. In particular, a missing
    # ThirdParty/Scotch shared library must never cause the Agent to rewrite a
    # valid snappyHexMesh case.
    if native_execution:
        native_commands = [item.command for item in build_graph.native_pipeline]
        if build_graph.surface_paths:
            native_commands.insert(0, "surfaceCheck")
        ready, toolchain_failures, toolchain_checked = self._native_toolchain_preflight(native_commands)
        if not ready:
            event = self._event(
                llm_step,
                "native_toolchain_preflight",
                False,
                "Selected OpenFOAM native toolchain is not runnable under the sanitized environment; no candidate case files were written and CFD repair was not invoked.",
                "\n".join(f"- {failure}" for failure in toolchain_failures),
                validation_status="inconclusive",
                failure_category="infra",
            )
            state.engineering_events.append(event)
            self._record_unresolved_failure(state, event)
            self._emit_engineering_event(
                f"{progress_phase}-case-build", event,
                step=progress_step, limit=progress_limit, state=state,
            )
            state.transition(
                State.ENGINEERING_BLOCKED,
                "OpenFOAM native toolchain preflight failed before case commit; "
                "source/fix the OpenFOAM ThirdParty/shared-library environment and retry. "
                + event.summary,
            )
            self._pending_candidate_execution = None
            self._pending_candidate_failed_paths = ()
            self._pending_execution_plan = None
            return True
    else:
        toolchain_checked = 0

    self.progress.emit(
        ProgressEvent(
            phase=f"{progress_phase}-case-build",
            message="controller-owned case build graph 검증 완료; transactional commit 시작",
            status="success",
            step=progress_step,
            limit=progress_limit,
            metrics={
                "requiredFiles": len(build_graph.required_paths),
                "authoredFiles": len(build_graph.authored_paths),
                "staticChecks": len(build_graph.dictionary_paths),
                "surfaceChecks": len(build_graph.surface_paths),
                "nativeCommands": len(build_graph.native_pipeline),
                "ignoredHints": len(build_graph.warnings),
                "reusedExistingFiles": len(reused_existing_paths),
                "toolchainChecked": toolchain_checked,
            },
        )
    )

    # Only after every candidate file passes deterministic authoring preflight do
    # we mutate the workspace. Commit the complete text bundle atomically at the
    # filesystem level; following write actions are audit/progress no-ops.
    self._pending_candidate_execution = None
    self._pending_candidate_failed_paths = ()
    # Since all file writes precede dictionary or
    # native execution in the expanded plan, later OpenFOAM failures always see a
    # complete authored bundle and can use true delta RepairTurn semantics.
    for path, content in rendered_files:
        actions.append(
            WriteCaseFileAction(
                type="write_case_file",
                path=path,
                content=content,
                rationale="",
            )
        )

    self._pending_execution_plan = execution.plan

    # v4.6: validation/native actions are compiled from the actual bundle. The
    # model may suggest a mesh strategy, but cannot independently schedule stale
    # per-file validators. Header/semantic checks already ran transactionally above.
    for path in build_graph.surface_paths:
        actions.append(
            SurfaceCheckAction(
                type="surface_check",
                path=path,
                rationale="controller-compiled surface validation",
            )
        )
    legacy_mesh_dispatch = {"blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "createPatch", "checkMesh"}
    for invocation in build_graph.native_pipeline:
        if invocation.command in legacy_mesh_dispatch and not invocation.arguments:
            actions.append(
                RunMeshCommandAction(
                    type="run_mesh_command",
                    command=invocation.command,
                    rationale="controller-compiled mesh/validation consumer",
                )
            )
        else:
            actions.append(
                RunNativeOpenFOAMAction(
                    type="run_openfoam_command",
                    invocation=invocation,
                )
            )
    actions.append(
        ValidatePreSolveAction(
            type="validate_pre_solve",
            required_case_files=list(build_graph.required_paths),
            rationale="controller-compiled required manifest",
        )
    )
    actions.append(
        FinishPreviewAction(
            type="finish_preview",
            plan=execution.plan,
            rationale="",
        )
    )

    execution_id = f"{progress_phase}:execution-plan:{llm_step:04d}"
    total = len(actions)
    remaining = self.policy.max_tool_actions - self._tool_action_count(state)
    if total > remaining:
        state.transition(
            State.ENGINEERING_BLOCKED,
            f"Engineering deterministic action budget cannot cover atomic case build graph ({total} needed, {remaining} remaining).",
        )
        return True
    committed = self.workspace.commit_text_transaction(candidate_bundle)
    self._precommitted_files.update(committed)
    self.progress.emit(
        ProgressEvent(
            phase=f"{progress_phase}-execution-plan",
            message=f"deterministic execution plan 시작: {execution.goal}",
            status="start",
            step=progress_step,
            limit=progress_limit,
            metrics={
                "actions": total,
                "files": len(build_graph.authored_paths),
                "requiredFiles": len(build_graph.required_paths),
                "surfaceChecks": len(build_graph.surface_paths),
                "nativeCommands": len(build_graph.native_pipeline),
                "controllerCompiled": True,
                "reusedExistingFiles": len(reused_existing_paths),
            },
        )
    )

    for index, member in enumerate(actions, start=1):
        if self._tool_action_count(state) >= self.policy.max_tool_actions:
            event = self._event(
                llm_step,
                getattr(member, "type", "unknown"),
                False,
                f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}); execution plan stopped.",
                validation_status="fail", failure_category="infra",
            )
            event = self._tag_execution_plan_event(
                event, execution, execution_id, index, total
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                f"{progress_phase}-execution-plan",
                event,
                step=index,
                limit=total,
                state=state,
            )
            state.transition(
                State.ENGINEERING_BLOCKED,
                f"Engineering deterministic action budget exhausted ({self.policy.max_tool_actions}).",
            )
            return True

        self._emit_action_started(
            f"{progress_phase}-execution-plan",
            member,
            step=index,
            limit=total,
        )
        event, terminal = self._dispatch_prepare(
            state,
            member,
            step=llm_step,
            native_execution=native_execution,
        )
        event = self._tag_execution_plan_event(
            event, execution, execution_id, index, total
        )
        state.engineering_events.append(event)
        self._emit_engineering_event(
            f"{progress_phase}-execution-plan",
            event,
            step=index,
            limit=total,
            state=state,
        )

        if not event.success:
            self.progress.emit(
                ProgressEvent(
                    phase=f"{progress_phase}-execution-plan",
                    message=f"execution plan 중단: {execution.goal}",
                    status="failure",
                    metrics={"executed": index, "planned": total},
                )
            )
            return self._route_failed_event(state, event, default_terminal=terminal)
        if isinstance(member, WriteCaseFileAction) and member.path == "system/blockMeshDict":
            self._structured_block_mesh = execution.block_mesh
        if terminal:
            self.progress.emit(
                ProgressEvent(
                    phase=f"{progress_phase}-execution-plan",
                    message=f"execution plan 완료: {execution.goal}",
                    status="success",
                    metrics={"executed": index, "planned": total},
                )
            )
            self._pending_execution_plan = None
            return True

    return False

