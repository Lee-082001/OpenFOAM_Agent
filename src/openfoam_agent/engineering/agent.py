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


@dataclass
class EngineeringPolicy:
    # Soft preparation budget. Reaching this boundary does not automatically
    # terminate the run: deterministic progress evidence can extend the window.
    max_agent_steps: int = 20
    hard_max_agent_steps: int = 40
    step_extension: int = 10
    progress_window: int = 8

    # Final plan submission is deliberately separated from tool work so a
    # successful checkMesh at a budget boundary cannot dead-end the run.
    max_finalization_steps: int = 3

    # Resource budgets are independent from LLM-turn budgets. Python only
    # bounds execution/retry cost; it does not make CFD design decisions.
    max_native_commands: int = 40
    max_mesh_repair_cycles: int = 6
    max_runtime_repair_steps: int = 10

    # v2.9: LLM-turn budgets and deterministic action budgets are separate. A single
    # LLM turn may authorize a full execution plan or a short bounded sequence.
    max_tool_actions: int = 160
    max_runtime_repair_tool_actions: int = 48

    # Complete-plan authoring failures happen before any case mutation. Keep this
    # retry class separately bounded so a model cannot burn the full Engineering
    # budget repeatedly regenerating an unsafe/unserializable bundle.
    max_case_plan_authoring_retries: int = 3

    # v2.13: retrieval is driven by explicit evidence gaps rather than free-form
    # repeated search turns. These are hard fuses, not the normal stopping rule;
    # novelty/stagnation is tracked per gap.
    max_prepare_retrieval_cycles: int = 3
    max_runtime_retrieval_cycles: int = 2

    observation_history: int = 12
    max_observation_chars: int = 12_000
    model_event_excerpt_chars: int = 2_500
    max_model_prompt_chars: int = 60_000
    # Authoring carries a frozen design plus file DSL and therefore needs a larger
    # context envelope than design/reasoning turns. The CLI intentionally keeps
    # design at 18k while allowing authoring 32k by default.
    max_authoring_prompt_chars: int = 32_000
    max_prepare_model_evidence_items: int = 16
    max_decide_model_evidence_items: int = 18
    max_model_evidence_detail_chars: int = 900
    max_new_model_evidence_per_batch: int = 6
    max_model_feedback_items: int = 8
    max_mesh_cells: int = 5_000_000
    require_solve_ready_gate: bool = False

    # v4.3 validation routing. Documentary dictionary probes are advisory;
    # stronger consumer validation can initialize the selected solver in a
    # Python-owned, serial, endTime=0 shadow case.
    foam_dictionary_probe: bool = False
    zero_step_consumer_validation: bool = True
    zero_step_validation_timeout: int = 60

    # v2.9: when the capability graph is small, preload deterministic provider
    # evidence into the first engineering prompt so solver selection does not
    # require an extra LLM -> search_capabilities -> LLM round trip.
    preload_capabilities: bool = False
    max_preloaded_capabilities: int = 24

    # v2.10 token controls. Kept opt-in at the library-policy level for API
    # compatibility; the production CLI enables both.
    compact_phase_schemas: bool = False
    state_delta_context: bool = False
    bounded_evidence_context: bool = False
    staged_case_authoring: bool = False
    isolation_policy_path: str | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "max_agent_steps": self.max_agent_steps,
            "hard_max_agent_steps": self.hard_max_agent_steps,
            "step_extension": self.step_extension,
            "progress_window": self.progress_window,
            "max_finalization_steps": self.max_finalization_steps,
            "max_native_commands": self.max_native_commands,
            "max_mesh_repair_cycles": self.max_mesh_repair_cycles,
            "max_runtime_repair_steps": self.max_runtime_repair_steps,
            "max_tool_actions": self.max_tool_actions,
            "max_runtime_repair_tool_actions": self.max_runtime_repair_tool_actions,
            "max_case_plan_authoring_retries": self.max_case_plan_authoring_retries,
            "max_prepare_retrieval_cycles": self.max_prepare_retrieval_cycles,
            "max_runtime_retrieval_cycles": self.max_runtime_retrieval_cycles,
            "observation_history": self.observation_history,
            "max_observation_chars": self.max_observation_chars,
            "model_event_excerpt_chars": self.model_event_excerpt_chars,
            "max_model_prompt_chars": self.max_model_prompt_chars,
            "max_authoring_prompt_chars": self.max_authoring_prompt_chars,
            "max_prepare_model_evidence_items": self.max_prepare_model_evidence_items,
            "max_decide_model_evidence_items": self.max_decide_model_evidence_items,
            "max_model_evidence_detail_chars": self.max_model_evidence_detail_chars,
            "max_new_model_evidence_per_batch": self.max_new_model_evidence_per_batch,
            "max_model_feedback_items": self.max_model_feedback_items,
            "max_mesh_cells": self.max_mesh_cells,
            "max_preloaded_capabilities": self.max_preloaded_capabilities,
            "zero_step_validation_timeout": self.zero_step_validation_timeout,
        }
        for name, value in integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.hard_max_agent_steps < self.max_agent_steps:
            raise ValueError("hard_max_agent_steps must be >= max_agent_steps")


@dataclass
class RepairOutcome:
    decision: RuntimeRepairDecision
    plan: EngineeringPlan | None = None
    reason: str = ""

    @property
    def retry(self) -> bool:
        """Backward-compatible convenience for the runtime orchestrator/tests."""

        return self.decision == RuntimeRepairDecision.RETRY_SOLVER


_MESH_TOPOLOGY_MUTATING_COMMANDS = frozenset({"blockMesh", "snappyHexMesh", "createPatch"})


class CFDEngineeringAgent:
    """Single agent that owns CFD design, implementation and failure repair."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        workspace: str | Path,
        capability_db: str | Path,
        tools: OpenFOAMTools | None = None,
        policy: EngineeringPolicy | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.llm = llm
        self.workspace = CaseWorkspace(workspace)
        self.tools = tools or OpenFOAMTools.for_workspace(self.workspace.root)
        self.catalog = CapabilityCatalog(capability_db, installation=getattr(self.tools, "installed_openfoam", None))
        self.references = OpenFOAMReferenceIndex()
        self.safety = DeterministicSafetyGate(self.tools, self.workspace)
        self.presolve = PreSolveCompletenessGate(self.tools, self.workspace)
        self.policy = policy or EngineeringPolicy()
        runner = getattr(self.tools, "runner", None)
        if isinstance(runner, SafeRunner):
            runner.budget.limit = self.policy.max_native_commands
            runner.resource_limits.max_native_processes = self.policy.max_native_commands
            if self.policy.isolation_policy_path:
                from openfoam_agent.tools.linux_isolation import LinuxIsolation, LinuxIsolationPolicy
                operator = LinuxIsolationPolicy.read(self.policy.isolation_policy_path)
                operator.limits.max_native_processes = min(operator.limits.max_native_processes, self.policy.max_native_commands)
                runner.resource_limits = operator.limits
                runner.budget.limit = operator.limits.max_native_processes
                runner.isolation = LinuxIsolation(operator, runner.workspace_root)
        self.progress = progress or NullProgressReporter()
        self._checkmesh_mesh_manifest: str | None = None
        self._presolve_case_manifest: str | None = None
        self._presolve_required_case_files: tuple[str, ...] | None = None
        self._pending_execution_plan: EngineeringPlan | None = None
        self._authoring_task_queue = None
        self._draft_design_plan: EngineeringPlan | None = None
        self._draft_authoring_brief: str = ""
        self._pending_candidate_execution: ExecuteCasePlanAction | None = None
        self._pending_candidate_failed_paths: tuple[str, ...] = ()
        self._structured_block_mesh: TypedBlockMeshFile | None = None
        self._phase_prompt_counts: dict[str, int] = {}
        self._phase_context_snapshots: dict[str, dict[str, str | None]] = {}
        self._evidence_gap_ledger: dict[str, dict[str, dict[str, object]]] = {}
        self._retrieval_cycles: dict[str, int] = {}
        self._evidence_retrieval_disabled: dict[str, str] = {}
        self._active_transaction = None
        self._precommitted_files: dict[str, str] = {}
        self._precommitted_drops: set[str] = set()
        self._resume_pending = False

    def prepare(self, state: CFDState, *, native_execution: bool = True) -> CFDState:
        from openfoam_agent.engineering.phases.lifecycle_controller import prepare
        return prepare(self, state, native_execution=native_execution)

    def revise_from_feedback(self, state: CFDState, *, native_execution: bool = True) -> CFDState:
        from openfoam_agent.engineering.phases.lifecycle_controller import revise_from_feedback
        return revise_from_feedback(self, state, native_execution=native_execution)

    def _begin_confirmed_revision_mutation(self, state: CFDState) -> None:
        from openfoam_agent.engineering.phases.lifecycle_controller import begin_confirmed_revision_mutation
        return begin_confirmed_revision_mutation(self, state)

    def _restore_unmutated_revision_ready(self, state: CFDState, proposal) -> None:
        from openfoam_agent.engineering.phases.lifecycle_controller import restore_unmutated_revision_ready
        return restore_unmutated_revision_ready(self, state, proposal)

    def _execute_prepare_decision(self, state, action, **kwargs):
        from openfoam_agent.engineering.phases.decision_controller import execute_prepare_decision
        return execute_prepare_decision(self, state, action, **kwargs)

    def _execute_prepare_decision_impl(self, state, action, *, llm_step: int, progress_phase: str, progress_step: int, progress_limit: int, native_execution: bool) -> bool:
        from openfoam_agent.engineering.phases.decision_controller import execute_prepare_decision_impl
        return execute_prepare_decision_impl(self, state, action, llm_step=llm_step, progress_phase=progress_phase, progress_step=progress_step, progress_limit=progress_limit, native_execution=native_execution)

    @staticmethod
    def _candidate_failure_paths(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.authoring_controller import candidate_failure_paths
        return candidate_failure_paths(self, *args, **kwargs)

    def _candidate_repair_context(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.authoring_controller import candidate_repair_context
        return candidate_repair_context(self, *args, **kwargs)

    def _apply_candidate_case_plan_repair(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.authoring_controller import apply_candidate_case_plan_repair
        return apply_candidate_case_plan_repair(self, *args, **kwargs)

    def _execute_candidate_block_mesh_repair(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import execute_candidate_block_mesh_repair
        return execute_candidate_block_mesh_repair(self, *args, **kwargs)

    def _execute_candidate_case_plan_repair(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import execute_candidate_case_plan_repair
        return execute_candidate_case_plan_repair(self, *args, **kwargs)

    @staticmethod
    def _failure_record(event: EngineeringEvent) -> dict[str, object]:
        return {
            "step": event.step,
            "action_type": event.action_type,
            "category": event.failure_category or "case",
            "validation_status": event.validation_status,
            "summary": event.summary,
            "output_excerpt": event.output_excerpt[:4000],
            "failure_signature": event.failure_signature,
        }

    def _record_unresolved_failure(self, state: CFDState, event: EngineeringEvent) -> None:
        record = self._failure_record(event)
        if state.primary_failure is None:
            state.primary_failure = record
            return
        prior = state.primary_failure
        if (
            prior.get("action_type") == record.get("action_type")
            and prior.get("summary") == record.get("summary")
            and prior.get("failure_signature") == record.get("failure_signature")
        ):
            return
        if len(state.secondary_failures) < 16:
            state.secondary_failures.append(record)

    def _route_failed_event(self, state: CFDState, event: EngineeringEvent, *, default_terminal: bool) -> bool:
        """Only semantic CASE failures are eligible for another CFD repair turn."""
        if event.success:
            return default_terminal
        self._record_unresolved_failure(state, event)
        category = event.failure_category or "case"
        if category in {"tool", "infra", "security", "user_contract"}:
            label = {
                "tool": "validation/tool infrastructure",
                "infra": "workflow/runtime infrastructure",
                "security": "security/execution policy",
                "user_contract": "confirmed user contract",
            }[category]
            state.transition(
                State.ENGINEERING_BLOCKED,
                f"Engineering stopped by {label} failure; CFD case repair was not invoked: {event.summary}",
            )
            return True
        return default_terminal

    @staticmethod
    def _clear_unresolved_failure(state: CFDState) -> None:
        state.primary_failure = None
        state.secondary_failures = []
        state.repair_episode = None

    def _execute_case_plan(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.authoring_controller import execute_case_plan
        return execute_case_plan(self, *args, **kwargs)

    def _execute_block_mesh_repair(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import execute_block_mesh_repair
        return execute_block_mesh_repair(self, *args, **kwargs)

    def _repair_actions(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import repair_actions
        return repair_actions(self, *args, **kwargs)

    def _execute_prepare_repair_plan(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import execute_prepare_repair_plan
        return execute_prepare_repair_plan(self, *args, **kwargs)

    def _execute_strategy_revision(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import execute_strategy_revision
        return execute_strategy_revision(self, *args, **kwargs)

    def _runtime_repair_actions(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import runtime_repair_actions
        return runtime_repair_actions(self, *args, **kwargs)

    def _execute_runtime_repair_plan(self, *args, **kwargs):
        from openfoam_agent.engineering.phases.repair_controller import execute_runtime_repair_plan
        return execute_runtime_repair_plan(self, *args, **kwargs)

    @staticmethod
    def _tag_execution_plan_event(
        event: EngineeringEvent,
        execution: ExecuteCasePlanAction,
        execution_id: str,
        index: int,
        total: int,
    ) -> EngineeringEvent:
        return event.model_copy(
            update={
                "sequence_id": execution_id,
                "sequence_goal": execution.goal,
                "sequence_index": index,
                "sequence_length": total,
            }
        )

    def _execute_prepare_sequence(
        self, state: CFDState, sequence: EngineeringSequenceAction, *, llm_step: int,
        progress_phase: str, progress_step: int, progress_limit: int, native_execution: bool
    ) -> bool:
        from openfoam_agent.engineering.phases.decision_controller import execute_prepare_sequence
        return execute_prepare_sequence(
            self, state, sequence, llm_step=llm_step, progress_phase=progress_phase,
            progress_step=progress_step, progress_limit=progress_limit, native_execution=native_execution
        )

    @staticmethod
    def _tag_sequence_event(
        event: EngineeringEvent,
        sequence: EngineeringSequenceAction,
        sequence_id: str,
        index: int,
    ) -> EngineeringEvent:
        return event.model_copy(
            update={
                "sequence_id": sequence_id,
                "sequence_goal": sequence.goal,
                "sequence_index": index,
                "sequence_length": len(sequence.actions),
            }
        )

    def _checkmesh_preflight(self, state: CFDState, *, phase: str) -> bool:
        """Fail fast before LLM engineering when trusted checkMesh is unavailable.

        This intentionally checks only checkMesh. Solver/mesh-strategy-specific tools remain
        agent-observed capabilities so their absence can be handled during engineering.
        """
        preflight = getattr(self.tools, "check_mesh_preflight", None)
        if not callable(preflight):
            # Injected test/dry tool doubles may not expose host executable discovery.
            return True
        status = preflight()
        available = bool(status.get("available")) and bool(status.get("trusted", True))
        if available:
            self.progress.emit(
                ProgressEvent(
                    phase=phase,
                    message="OpenFOAM native preflight: checkMesh 확인",
                    status="success",
                    metrics={"checkMesh": "available"},
                )
            )
            return True

        reason = str(status.get("reason") or "trusted checkMesh executable is unavailable")
        state.engineering_round_start_index = len(state.engineering_events)
        state.transition(
            State.ENGINEERING_BLOCKED,
            "OpenFOAM native preflight failed before autonomous engineering: "
            f"checkMesh is unavailable or untrusted. {reason} "
            "Source the OpenFOAM environment before launching/restarting openfoam-agent.",
        )
        self.progress.emit(
            ProgressEvent(
                phase=phase,
                message="OpenFOAM native preflight: checkMesh를 찾을 수 없음",
                status="failure",
                metrics={"engineeringActions": 0},
            )
        )
        return False

    def _progress_allows_extension(self, state: CFDState) -> tuple[bool, str]:
        """Decide only whether more bounded work is justified, never what CFD work to do."""

        events = self._current_round_events(state)
        window = self.policy.progress_window
        if len(events) < window:
            return False, "insufficient recent history to prove progress"

        recent = events[-window:]
        earlier = events[:-window]
        earlier_signatures = {self._progress_signature(event) for event in earlier}

        evidence_actions = {
            "search_capabilities",
            "search_references",
            "read_reference",
            "gather_evidence",
            "write_case_file",
            "delete_case_file",
            "validate_dictionary",
            "surface_check",
            "run_mesh_command",
        }
        recent_evidence = [
            event
            for event in recent
            if event.action_type in evidence_actions
            and (event.action_type != "gather_evidence" or bool(event.observed_evidence))
        ]
        novel = [
            event
            for event in recent_evidence
            if self._progress_signature(event) not in earlier_signatures
        ]
        if not novel:
            return False, "recent actions only repeated previously observed action/result signatures"

        # A new case artifact, new tool/reference observation, or changed native
        # result is enough to justify another bounded chunk. This intentionally
        # does not judge whether the CFD choice itself is good.
        kinds = sorted({event.action_type for event in novel})
        return True, f"new evidence observed via {', '.join(kinds)}"

    @staticmethod
    def _progress_signature(event: EngineeringEvent) -> str:
        output = re.sub(
            r"(?im)^\s*(?:ExecutionTime|ClockTime)\b.*$",
            "",
            event.output_excerpt,
        )
        payload = "\n".join(
            [
                event.action_type,
                "1" if event.success else "0",
                event.summary,
                output,
                event.artifact_sha256 or "",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _current_round_events(state: CFDState) -> list[EngineeringEvent]:
        start = min(state.engineering_round_start_index, len(state.engineering_events))
        return state.engineering_events[start:]

    def checkpoint(self, state, reason):
        from openfoam_agent.workflow.checkpoint import CheckpointStore
        return CheckpointStore(self).save(state, reason=reason)

    def bind_checkpoint(self, state):
        runner = getattr(self.tools, "runner", None)
        if isinstance(runner, SafeRunner):
            runner.process_observer = lambda phase, record: self.checkpoint(state, f"native-{phase}")

    def restore_checkpoint(self, *, reconcile_parallel_restart=False):
        from openfoam_agent.workflow.checkpoint import CheckpointStore
        state = CheckpointStore(self).load(reconcile_parallel_restart=reconcile_parallel_restart)
        self.bind_checkpoint(state)
        return state

    def _invalidate_mesh_dependencies(self, state):
        if state is None:
            return
        from openfoam_agent.contracts.regions import region_mesh_digest
        stale = [name for name, old in state.region_mesh_manifests.items()
                 if region_mesh_digest(self.workspace, name) != old]
        if stale:
            for name in stale:
                state.region_mesh_evidence.pop(name, None)
                state.region_mesh_manifests.pop(name, None)
            state.mesh_evidence = None
            state.case_seal = None
            state.solve_approved = False
            state.execution_approval = None
            self._checkmesh_mesh_manifest = None
            self._presolve_case_manifest = None

    def _record_region_mesh(self, state, evidence, arguments):
        region = ""
        if "-region" in arguments:
            pos = arguments.index("-region")
            if pos + 1 >= len(arguments):
                raise ValueError("Missing region argument.")
            region = arguments[pos + 1]
        state.region_mesh_evidence[region] = evidence
        if evidence.passed:
            state.region_mesh_manifests[region] = region_mesh_digest(self.workspace, region)
        else:
            state.region_mesh_manifests.pop(region, None)
        state.mesh_evidence = evidence

    def _region_mesh_failures(self, state, plan):
        failures = []
        for layout in region_layouts(plan):
            if not layout.region:
                continue
            evidence = state.region_mesh_evidence.get(layout.region)
            if evidence is None or not evidence.passed:
                failures.append(f"Passing checkMesh evidence is missing for region {layout.region}.")
            elif state.region_mesh_manifests.get(layout.region) != region_mesh_digest(self.workspace, layout.region):
                failures.append(f"Mesh evidence is stale for region {layout.region}.")
        return failures

    def _native_command_count(self, state: CFDState) -> int:
        runner = getattr(self.tools, "runner", None)
        if isinstance(runner, SafeRunner):
            state.native_process_records = list(runner.budget.records)
            return runner.budget.attempts
        # Injected test transports have no OS process counter. Do not label these
        # action counts as measured subprocesses in reports.
        return sum(1 for event in self._current_round_events(state) if event.native_command_executed)

    def _tool_action_count(self, state: CFDState) -> int:
        return len(self._current_round_events(state))

    def _mesh_repair_status(self, state: CFDState) -> tuple[int, bool, bool]:
        """Return (completed/started repair cycles, failure pending, current cycle started)."""

        cycles = 0
        failure_pending = False
        current_cycle_started = False
        for event in self._current_round_events(state):
            if event.mesh_command_executed:
                if event.success:
                    failure_pending = False
                    current_cycle_started = False
                else:
                    failure_pending = True
                    current_cycle_started = False
                continue
            if (
                failure_pending
                and event.success
                and event.action_type in {"write_case_file", "patch_case_file", "delete_case_file"}
                and not current_cycle_started
            ):
                cycles += 1
                current_cycle_started = True
        return cycles, failure_pending, current_cycle_started

    def _mesh_repair_cycle_count(self, state: CFDState) -> int:
        return self._mesh_repair_status(state)[0]

    def _ready_for_finalization(self, state: CFDState, *, native_execution: bool) -> bool:
        # Do not silently grant extra retries after a rejected terminal claim.  The
        # finalization window exists only for the boundary case where ordinary tool
        # work consumed the budget and the case itself is already current/validated.
        if state.engineering_events and state.engineering_events[-1].action_type in {
            "finish_preview",
            "block",
            "retry_solver",
        }:
            return False
        if not native_execution:
            return bool(self.workspace.file_seals())
        evidence = state.mesh_evidence
        return bool(
            evidence is not None
            and evidence.passed
            and (evidence.cell_count is None or evidence.cell_count <= self.policy.max_mesh_cells)
            and self._checkmesh_mesh_manifest is not None
            and self._checkmesh_mesh_manifest == self.workspace.mesh_manifest_digest()
        )

    def _run_finalization_window(
        self,
        state: CFDState,
        *,
        native_execution: bool,
        start_step: int,
        phase: str = "prepare_finalize",
    ) -> CFDState:
        for offset in range(self.policy.max_finalization_steps):
            step = start_step + offset
            turn = self._generate_turn(
                state,
                step=step,
                local_step=offset + 1,
                current_step_limit=self.policy.max_finalization_steps,
                phase=phase,
                native_execution=native_execution,
            )
            action = turn.action
            progress_phase = "revision-finalize" if phase.startswith("human_revision") else "finalizing"
            self._emit_action_started(
                progress_phase, action, step=offset + 1, limit=self.policy.max_finalization_steps
            )

            if isinstance(action, (FinishPreviewAction, BlockAction)):
                event, terminal = self._dispatch_prepare(
                    state,
                    action,
                    step=step,
                    native_execution=native_execution,
                )
            else:
                event = self._event(
                    step,
                    getattr(action, "type", "unknown"),
                    False,
                    "Engineering tool budget is exhausted; finalization window permits only finish_preview or block.",
                )
                terminal = False

            state.engineering_events.append(event)
            self._emit_engineering_event(
                progress_phase,
                event,
                step=offset + 1,
                limit=self.policy.max_finalization_steps,
                state=state,
            )
            if terminal:
                return state

        state.transition(
            State.ENGINEERING_BLOCKED,
            f"Engineering finalization budget exhausted ({self.policy.max_finalization_steps}) after the case was validated.",
        )
        return state

    def _runtime_repair_outcome(
        self,
        state: CFDState,
        decision: RuntimeRepairDecision,
        reason: str,
        plan: EngineeringPlan | None = None,
    ) -> RepairOutcome:
        """Close the transient RUNTIME_REPAIR state before returning to the workflow.

        RUNTIME_REPAIR is an internal state owned by RuntimeOrchestrator/Engineering. It
        must never leak back to the top-level workflow. The decision enum makes every
        non-retry exit explicit instead of overloading a boolean.
        """

        if decision == RuntimeRepairDecision.RETRY_SOLVER:
            if state.current_state != State.SIMULATION:
                state.transition(State.SIMULATION, reason or "Runtime repair requested solver retry.")
        elif decision == RuntimeRepairDecision.NEEDS_USER_REVIEW:
            state.solve_approved = False
            state.transition(State.ENGINEERING_REVIEW_REQUIRED, reason)
        elif decision == RuntimeRepairDecision.STRATEGY_REVISION:
            state.solve_approved = False
            state.transition(State.ENGINEERING_REVIEW_REQUIRED, reason)
        else:
            state.solve_approved = False
            state.transition(State.ENGINEERING_BLOCKED, reason)
        return RepairOutcome(decision=decision, plan=plan, reason=reason)

    def repair_runtime(
        self,
        state: CFDState,
        *,
        runtime_log: str,
        attempt: int,
        native_execution: bool = True,
    ) -> RepairOutcome:
        if state.engineering_plan is None or state.case_seal is None:
            return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, "No approved engineering plan is available.")
        self._repair_review_required = None
        if not state.solve_approved or state.execution_approval is None:
            return self._runtime_repair_outcome(state, RuntimeRepairDecision.NEEDS_USER_REVIEW, "Runtime repair requires a durable execution approval.")
        try:
            state.execution_approval.check_plan(state.engineering_plan)
        except ValueError as exc:
            return self._runtime_repair_outcome(state, RuntimeRepairDecision.NEEDS_USER_REVIEW, str(exc))
        try:
            self.workspace.adopt_seal(state.case_seal)
            self.safety.verify_seal(state.engineering_plan, state.case_seal)
        except WorkspaceSafetyError as exc:
            return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, f"Approved case integrity verification failed: {exc}")
        if state.mesh_evidence is not None and state.mesh_evidence.passed:
            # The full case seal proves the runtime-repair workspace is still the exact
            # case for which this persisted checkMesh evidence was approved. Restore
            # freshness against the narrower mesh-only manifest so solver-input edits
            # do not force redundant checkMesh runs.
            observed = state.region_mesh_manifests.get("")
            if observed is not None and observed == region_mesh_digest(self.workspace, ""):
                self._checkmesh_mesh_manifest = self.workspace.mesh_manifest_digest()
        approved_solver = state.engineering_plan.solver
        self._evidence_gap_ledger["runtime_repair"] = {}
        self._retrieval_cycles["runtime_repair"] = 0
        self._evidence_retrieval_disabled.pop("runtime_repair", None)
        state.last_runtime_log_excerpt = runtime_log[-12000:]
        state.transition(
            State.RUNTIME_REPAIR,
            f"foamRun attempt {attempt} failed; log returned to CFDEngineeringAgent.",
        )
        self.progress.emit(
            ProgressEvent(
                phase="runtime-repair",
                message=f"runtime repair cycle 시작: solver attempt={attempt}",
                status="start",
                metrics={
                    "llmTurnBudget": self.policy.max_runtime_repair_steps,
                    "toolBudget": self.policy.max_runtime_repair_tool_actions,
                },
            )
        )
        runtime_event_start = len(state.engineering_events)
        for local_step in range(1, self.policy.max_runtime_repair_steps + 1):
            step = len(state.engineering_events) + 1
            turn = self._generate_turn(
                state,
                step=step,
                local_step=local_step,
                current_step_limit=self.policy.max_runtime_repair_steps,
                phase="runtime_repair",
                runtime_log=runtime_log,
                native_execution=native_execution,
            )
            action = turn.action
            if isinstance(action, (RuntimeCaseRepairAction, RepairCasePlanAction)):
                outcome = self._execute_runtime_repair_plan(
                    state,
                    action,
                    approved_solver=approved_solver,
                    llm_step=step,
                    native_execution=native_execution,
                    runtime_event_start=runtime_event_start,
                )
                if outcome is not None:
                    return self._runtime_repair_outcome(
                        state, outcome.decision, outcome.reason, outcome.plan
                    )
                continue
            if isinstance(action, EngineeringSequenceAction):
                outcome = self._execute_runtime_sequence(
                    state,
                    action,
                    approved_solver=approved_solver,
                    llm_step=step,
                    progress_step=local_step,
                    native_execution=native_execution,
                    runtime_event_start=runtime_event_start,
                )
                if outcome is not None:
                    return self._runtime_repair_outcome(
                        state, outcome.decision, outcome.reason, outcome.plan
                    )
                continue

            if len(state.engineering_events) - runtime_event_start >= self.policy.max_runtime_repair_tool_actions:
                reason = (
                    "Runtime repair deterministic action budget exhausted "
                    f"({self.policy.max_runtime_repair_tool_actions})."
                )
                return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, reason)

            self._emit_action_started(
                "runtime-repair",
                action,
                step=local_step,
                limit=self.policy.max_runtime_repair_steps,
            )
            if isinstance(action, RetrySolverAction):
                event, outcome = self._dispatch_retry_solver(
                    state,
                    action,
                    approved_solver=approved_solver,
                    step=step,
                    native_execution=native_execution,
                )
                state.engineering_events.append(event)
                self._emit_engineering_event(
                    "runtime-repair",
                    event,
                    step=local_step,
                    limit=self.policy.max_runtime_repair_steps,
                    state=state,
                )
                if outcome is not None:
                    return self._runtime_repair_outcome(
                        state, outcome.decision, outcome.reason, outcome.plan
                    )
                continue
            if isinstance(action, BlockAction):
                event = self._event(step, action.type, True, action.reason)
                state.engineering_events.append(event)
                self._emit_engineering_event(
                    "runtime-repair",
                    event,
                    step=local_step,
                    limit=self.policy.max_runtime_repair_steps,
                    state=state,
                )
                decision = (
                    RuntimeRepairDecision.NEEDS_USER_REVIEW
                    if action.needs_user_input
                    else RuntimeRepairDecision.BLOCKED
                )
                return self._runtime_repair_outcome(state, decision, action.reason)

            event = self._dispatch_tool_action(
                action,
                step=step,
                native_execution=native_execution,
                phase="runtime_repair",
                state=state,
            )
            state.engineering_events.append(event)
            self._emit_engineering_event(
                "runtime-repair",
                event,
                step=local_step,
                limit=self.policy.max_runtime_repair_steps,
                state=state,
            )
            if getattr(self, "_repair_review_required", None):
                return self._runtime_repair_outcome(state, RuntimeRepairDecision.NEEDS_USER_REVIEW, self._repair_review_required)
        reason = f"Runtime repair action budget exhausted ({self.policy.max_runtime_repair_steps})."
        return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, reason)

    def _execute_runtime_sequence(
        self,
        state: CFDState,
        sequence: EngineeringSequenceAction,
        *,
        approved_solver: str,
        llm_step: int,
        progress_step: int,
        native_execution: bool,
        runtime_event_start: int,
    ) -> RepairOutcome | None:
        sequence_id = f"runtime-repair:{llm_step:04d}"
        self.progress.emit(
            ProgressEvent(
                phase="runtime-repair-sequence",
                message=f"sequence 시작: {sequence.goal}",
                status="start",
                step=progress_step,
                limit=self.policy.max_runtime_repair_steps,
                metrics={"actions": len(sequence.actions)},
            )
        )
        for index, member in enumerate(sequence.actions, start=1):
            if len(state.engineering_events) - runtime_event_start >= self.policy.max_runtime_repair_tool_actions:
                reason = (
                    "Runtime repair deterministic action budget exhausted "
                    f"({self.policy.max_runtime_repair_tool_actions})."
                )
                return self._runtime_repair_outcome(state, RuntimeRepairDecision.BLOCKED, reason)

            self._emit_action_started(
                "runtime-repair-sequence",
                member,
                step=index,
                limit=len(sequence.actions),
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
            elif isinstance(member, FinishPreviewAction):
                event = self._event(
                    llm_step,
                    member.type,
                    False,
                    "finish_preview is not valid inside runtime repair; use retry_solver as the sequence terminator.",
                )
            else:
                event = self._dispatch_tool_action(
                    member,
                    step=llm_step,
                    native_execution=native_execution,
                    phase="runtime_repair",
                    state=state,
                )
            event = self._tag_sequence_event(event, sequence, sequence_id, index)
            state.engineering_events.append(event)
            self._emit_engineering_event(
                "runtime-repair-sequence",
                event,
                step=index,
                limit=len(sequence.actions),
                state=state,
            )
            if not event.success:
                if getattr(self, "_repair_review_required", None):
                    return RepairOutcome(RuntimeRepairDecision.NEEDS_USER_REVIEW, reason=self._repair_review_required)
                self.progress.emit(
                    ProgressEvent(
                        phase="runtime-repair-sequence",
                        message=f"sequence 중단: {sequence.goal}",
                        status="failure",
                        metrics={"executed": index, "planned": len(sequence.actions)},
                    )
                )
                return outcome
            if outcome is not None:
                self.progress.emit(
                    ProgressEvent(
                        phase="runtime-repair-sequence",
                        message=f"sequence 완료: {sequence.goal}",
                        status="success",
                        metrics={"executed": index, "planned": len(sequence.actions)},
                    )
                )
                return outcome

        self.progress.emit(
            ProgressEvent(
                phase="runtime-repair-sequence",
                message=f"sequence 완료: {sequence.goal}",
                status="success",
                metrics={"executed": len(sequence.actions)},
            )
        )
        return None

    def _dispatch_retry_solver(
        self,
        state: CFDState,
        action: RetrySolverAction,
        *,
        approved_solver: str,
        step: int,
        native_execution: bool,
    ) -> tuple[EngineeringEvent, RepairOutcome | None]:
        result = self.safety.validate_plan(action.plan, state.intake)  # type: ignore[arg-type]
        if native_execution:
            result.failures.extend(self._region_mesh_failures(state, action.plan))
            result.valid = not result.failures
        result.failures.extend(self._validate_observed_provenance(action.plan, state))
        result.failures.extend(self._validate_engineering_defaults(action.plan, state))
        result.valid = not result.failures
        if state.execution_approval is not None:
            try:
                state.execution_approval.check_plan(action.plan)
                candidate = self.workspace.seal(action.plan)
                state.execution_approval.check_repair_files(candidate)
            except (ValueError, WorkspaceSafetyError) as exc:
                result.failures.append(str(exc))
                result.valid = False
        if action.plan.solver != approved_solver:
            result.failures.append("Runtime repair attempted to change the user-approved solver.")
            result.valid = False
        if native_execution and result.valid:
            native = self.safety.validate_native_inputs()
            result.failures.extend(native.failures)
            if self.policy.require_solve_ready_gate:
                presolve_is_current = bool(
                    self._presolve_case_manifest == self.workspace.manifest_digest()
                    and self._presolve_required_case_files
                    == tuple(action.plan.required_case_files)
                )
                if not presolve_is_current:
                    presolve = self.presolve.validate(action.plan)
                    result.failures.extend(presolve.failures)
                    if presolve.valid:
                        self._presolve_case_manifest = self.workspace.manifest_digest()
                        self._presolve_required_case_files = tuple(action.plan.required_case_files)
            if state.mesh_evidence is None or not state.mesh_evidence.passed:
                result.failures.append(
                    "A passing checkMesh result with cell-count evidence is required before an automatic solver retry."
                )
            elif state.mesh_evidence.cell_count is not None and state.mesh_evidence.cell_count > self.policy.max_mesh_cells:
                result.failures.append(
                    f"Mesh cell count {state.mesh_evidence.cell_count} exceeds bounded policy limit {self.policy.max_mesh_cells}."
                )
            elif self._checkmesh_mesh_manifest != self.workspace.mesh_manifest_digest():
                result.failures.append(
                    "Mesh-affecting inputs changed after checkMesh; re-run checkMesh before retry_solver."
                )
            result.valid = not result.failures
        if result.valid:
            state.engineering_plan = action.plan
            state.case_seal = self.workspace.seal(action.plan)
            state.transition(State.SIMULATION, "Engineering repair requested bounded solver retry.")
            return (
                self._event(
                    step,
                    action.type,
                    True,
                    "Runtime repair validated, pre-solve complete, and sealed; solver retry requested.",
                ),
                RepairOutcome(RuntimeRepairDecision.RETRY_SOLVER, action.plan),
            )
        return (
            self._event(
                step,
                action.type,
                False,
                "Solver retry was rejected by deterministic safety/pre-solve validation.",
                "\n".join(result.failures),
            ),
            None,
        )

    def _dispatch_prepare(self, state: CFDState, action, *, step: int, native_execution: bool):
        from openfoam_agent.engineering.phases.decision_controller import dispatch_prepare
        return dispatch_prepare(self, state, action, step=step, native_execution=native_execution)

    @staticmethod
    def _next_evidence_gap_id(*, occupied: set[str]) -> str:
        """Issue a deterministic opaque gap ID not already owned by the ledger.

        Gap IDs are workflow bookkeeping only.  The Agent may propose one, but Python
        owns identity/lifecycle so a harmless model-side collision cannot terminate CFD.
        """

        numbers = sorted(
            int(match.group(1))
            for item in occupied
            if (match := re.fullmatch(r"G([0-9]{2,4})", str(item))) is not None
        )
        start = (numbers[-1] + 1) if numbers and numbers[-1] < 9999 else 1
        for offset in range(9999):
            number = ((start - 1 + offset) % 9999) + 1
            candidate = f"G{number:02d}"
            if candidate not in occupied:
                return candidate
        raise RuntimeError("Evidence-gap ID space exhausted.")

    @staticmethod
    def _evidence_gap_protocol_fingerprint(gap: EvidenceGapRequest) -> str:
        payload = gap.model_dump(
            mode="json",
            exclude={"gap_id", "refines_gap_id"},
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]

    def _normalize_evidence_gap_batch(
        self,
        action: GatherEvidenceAction,
        *,
        phase: str,
    ) -> list[tuple[EvidenceGapRequest, str, list[str]]]:
        """Resolve harmless LLM gap-ID mistakes against the authoritative phase ledger.

        The model chooses *what evidence is needed*. Python owns opaque IDs and the
        refinement lifecycle.  This keeps protocol metadata from becoming a CFD-fatal
        structured-output error while preserving the single-retrieval hard fuse.
        """

        ledger = self._evidence_gap_ledger.setdefault(phase, {})
        occupied = set(ledger)
        reserved: set[str] = set()
        exact_seen: set[tuple[str, str | None, str]] = set()
        normalized: list[tuple[EvidenceGapRequest, str, list[str]]] = []

        for gap in action.gaps:
            requested_id = gap.gap_id
            parent_id = gap.refines_gap_id
            fingerprint = self._evidence_gap_protocol_fingerprint(gap)
            exact_key = (requested_id, parent_id, fingerprint)
            if exact_key in exact_seen:
                # Exact duplicate in one model response is pure protocol noise.
                continue
            exact_seen.add(exact_key)

            effective_id = requested_id
            notes: list[str] = []

            if parent_id == requested_id:
                if requested_id in ledger:
                    # The common model mistake seen in v3.3: "refine G1086" while
                    # reusing G1086 as the child. Preserve the intended parent and
                    # issue a fresh child ID.
                    effective_id = self._next_evidence_gap_id(
                        occupied=occupied | reserved
                    )
                    notes.append(
                        f"self-refining requested ID {requested_id} reissued as {effective_id}"
                    )
                else:
                    # No such parent exists, so the self-reference carries no usable
                    # lifecycle meaning. Keep the proposed child and clear the parent.
                    parent_id = None
                    notes.append(
                        f"self-refinement parent {requested_id} was unknown and was cleared"
                    )

            if effective_id in reserved:
                reissued = self._next_evidence_gap_id(occupied=occupied | reserved)
                notes.append(
                    f"duplicate in-batch requested ID {effective_id} reissued as {reissued}"
                )
                effective_id = reissued
            elif effective_id in ledger and parent_id is not None and effective_id != parent_id:
                # A refinement child collided with some already-owned ID.  Reissue the
                # child without changing its requested evidence semantics.
                reissued = self._next_evidence_gap_id(occupied=occupied | reserved)
                notes.append(
                    f"colliding requested ID {effective_id} reissued as {reissued}"
                )
                effective_id = reissued

            if parent_id is not None and parent_id not in ledger:
                notes.append(
                    f"unknown refinement parent {parent_id} cleared; request treated as a root gap"
                )
                parent_id = None

            normalized_gap = gap.model_copy(
                update={
                    "gap_id": effective_id,
                    "refines_gap_id": parent_id,
                }
            )
            normalized.append((normalized_gap, requested_id, notes))
            reserved.add(effective_id)

        return normalized

    def _evidence_gap_status(self, phase: str) -> list[dict[str, object]]:
        ledger = self._evidence_gap_ledger.get(phase, {})
        result: list[dict[str, object]] = []
        for gap_id, item in sorted(ledger.items()):
            seen = item.get("seen_ids", set())
            result.append(
                {
                    "gap_id": gap_id,
                    "missing_evidence": item.get("missing_evidence", ""),
                    "why_required": item.get("why_required", ""),
                    "retrievals": int(item.get("retrievals", 0)),
                    "seen_evidence_count": len(seen) if isinstance(seen, set) else len(list(seen)),
                    "last_new_evidence_count": int(item.get("last_new_count", 0)),
                    "status": str(item.get("status", "open")),
                    "refines_gap_id": item.get("refines_gap_id"),
                    "superseded_by": item.get("superseded_by"),
                    "stagnant": str(item.get("status", "")) == "stagnant",
                    "satisfied": str(item.get("status", "")) == "satisfied",
                }
            )
        return result

    def _mark_evidence_gaps_satisfied(self, phase: str) -> None:
        """Close retrieved gaps when the Agent elects to proceed without more retrieval."""

        ledger = self._evidence_gap_ledger.get(phase, {})
        for entry in ledger.values():
            if str(entry.get("status", "")) == "evidence_available":
                entry["status"] = "satisfied"

    def _store_evidence_payload(
        self,
        state: CFDState,
        *,
        phase: str,
        step: int,
        action_type: str,
        payload: object,
        observed_evidence: list[ObservedEngineeringEvidence],
    ) -> str:
        """Persist structured retrieval payload outside the progress event log."""

        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(
            f"{state.run_id}\0{phase}\0{step}\0{action_type}\0{canonical}".encode("utf-8")
        ).hexdigest()[:20]
        record_id = f"evrec_{digest}"
        record = EngineeringEvidenceRecord(
            record_id=record_id,
            phase=phase,
            step=step,
            action_type=action_type,
            payload=payload,
            observed_evidence=list(observed_evidence),
        )
        for index, existing in enumerate(state.engineering_evidence_records):
            if existing.record_id == record_id:
                state.engineering_evidence_records[index] = record
                break
        else:
            state.engineering_evidence_records.append(record)
        return record_id

    @staticmethod
    def _evidence_batch_display(gap_results: list[dict[str, object]], *, cycle: int, limit: int) -> str:
        """Return compact protocol metadata only; full evidence stays in the structured store."""

        compact_gaps: list[dict[str, object]] = []
        for item in gap_results:
            compact_gaps.append(
                {
                    key: item[key]
                    for key in (
                        "gap_id",
                        "requested_gap_id",
                        "status",
                        "new_evidence_ids",
                        "projected_evidence_ids",
                        "total_seen",
                        "message",
                        "protocol_notes",
                    )
                    if key in item
                }
            )
        return json.dumps(
            {"cycle": cycle, "cycle_limit": limit, "gaps": compact_gaps},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _disable_evidence_retrieval(self, phase: str, reason: str) -> None:
        """Fail closed for retrieval infrastructure without trapping the LLM in retry loops."""

        self._evidence_retrieval_disabled[phase] = reason[:1200]
        limit = (
            self.policy.max_runtime_retrieval_cycles
            if phase == "runtime_repair"
            else self.policy.max_prepare_retrieval_cycles
        )
        self._retrieval_cycles[phase] = limit
        for entry in self._evidence_gap_ledger.get(phase, {}).values():
            if str(entry.get("status", "")) in {"open", "evidence_available", "stagnant"}:
                entry["status"] = "retrieval_unavailable"

    @staticmethod
    def _evidence_relevance_score(record: dict[str, object]) -> int:
        """Deterministically rank retrieved evidence before projecting it to the LLM.

        Search tools may return many candidates.  All candidates remain in the durable
        evidence payload/ledger, but only a small high-value subset should be promoted
        into the next model context.  Native/runtime-strength capability evidence and
        full reference excerpts outrank shallow search snippets.
        """

        kind = str(record.get("kind", ""))
        score = 0
        if kind == "openfoam_reference":
            if str(record.get("content_excerpt", "")).strip():
                score += 80
            elif str(record.get("snippet", "")).strip():
                score += 35
            haystack = " ".join(
                str(record.get(key, "")) for key in ("reference", "snippet", "content_excerpt")
            )
        else:
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            level = str(result.get("verification_level", "")).casefold()
            level_weight = {
                "native": 90,
                "runtime": 85,
                "runtime_table": 80,
                "installed_binary": 70,
                "installed_library": 65,
                "installed": 55,
                "source": 35,
                "documented": 20,
            }
            score += level_weight.get(level, 0)
            if result.get("verified") is True:
                score += 25
            haystack = " ".join(
                str(result.get(key, "")) for key in ("name", "provider_id", "provider_type", "summary")
            ) + " " + str(record.get("reference", ""))

        query = str(record.get("query", ""))
        query_tokens = {tok.casefold() for tok in re.findall(r"[A-Za-z0-9_]{3,}", query)}
        hay_tokens = {tok.casefold() for tok in re.findall(r"[A-Za-z0-9_]{3,}", haystack)}
        score += min(30, 5 * len(query_tokens & hay_tokens))
        return score

    def _select_new_evidence_for_model(
        self,
        gap_results: list[dict[str, object]],
        records: dict[str, dict[str, object]],
    ) -> list[str]:
        """Fairly select a bounded subset of newly retrieved evidence for model context."""

        limit = self.policy.max_new_model_evidence_per_batch
        ranked_per_gap: list[list[str]] = []
        for gap in gap_results:
            ids = [str(item) for item in gap.get("new_evidence_ids", []) if str(item) in records]
            ids.sort(key=lambda eid: (-self._evidence_relevance_score(records[eid]), eid))
            if ids:
                ranked_per_gap.append(ids)

        promoted: list[str] = []
        cursor = 0
        while len(promoted) < limit and ranked_per_gap:
            next_round: list[list[str]] = []
            for queue in ranked_per_gap:
                if len(promoted) >= limit:
                    break
                if cursor < len(queue):
                    promoted.append(queue[cursor])
                    if cursor + 1 < len(queue):
                        next_round.append(queue)
            if not next_round:
                break
            ranked_per_gap = next_round
            cursor += 1
        return promoted

    def _gather_evidence(
        self,
        action: GatherEvidenceAction,
        *,
        step: int,
        phase: str,
        state: CFDState,
    ) -> EngineeringEvent:
        """Resolve explicit evidence gaps with bounded deterministic batch retrieval."""

        limit = (
            self.policy.max_runtime_retrieval_cycles
            if phase == "runtime_repair"
            else self.policy.max_prepare_retrieval_cycles
        )
        cycles = self._retrieval_cycles.get(phase, 0)
        disabled_reason = self._evidence_retrieval_disabled.get(phase)
        if disabled_reason:
            return self._event(
                step,
                action.type,
                False,
                f"Evidence retrieval is disabled for {phase} after an infrastructure failure; use existing evidence/defaults or block.",
                disabled_reason,
                failure_signature=f"evidence_retrieval:{phase}:disabled",
                failure_scope="pipeline",
            )
        if cycles >= limit:
            return self._event(
                step,
                action.type,
                False,
                f"Evidence retrieval hard fuse reached for {phase} ({limit} cycle(s)); use existing evidence/defaults or block.",
            )
        ledger = self._evidence_gap_ledger.setdefault(phase, {})
        performed_retrieval = False
        observed_by_id: dict[str, ObservedEngineeringEvidence] = {}
        new_candidate_records: dict[str, dict[str, object]] = {}
        gap_results: list[dict[str, object]] = []

        normalized_gaps = self._normalize_evidence_gap_batch(action, phase=phase)
        for gap, requested_gap_id, protocol_notes in normalized_gaps:
            existing = ledger.get(gap.gap_id)
            if existing is not None:
                result: dict[str, object] = {
                    "gap_id": gap.gap_id,
                    "status": "already_retrieved_blocked",
                    "message": (
                        "Each evidence gap may be retrieved once. Use the accumulated evidence, "
                        "proceed, or declare a more-specific refinement; Python owns the child ID."
                    ),
                }
                if requested_gap_id != gap.gap_id:
                    result["requested_gap_id"] = requested_gap_id
                if protocol_notes:
                    result["protocol_notes"] = protocol_notes
                gap_results.append(result)
                continue

            parent = None
            if gap.refines_gap_id is not None:
                parent = ledger.get(gap.refines_gap_id)
                if parent is not None and str(parent.get("status", "")) == "satisfied":
                    result = {
                        "gap_id": gap.gap_id,
                        "status": "satisfied_parent_blocked",
                        "message": f"Evidence gap {gap.refines_gap_id} is already satisfied and cannot be refined.",
                    }
                    if requested_gap_id != gap.gap_id:
                        result["requested_gap_id"] = requested_gap_id
                    if protocol_notes:
                        result["protocol_notes"] = protocol_notes
                    gap_results.append(result)
                    continue

            entry = {
                "missing_evidence": gap.missing_evidence,
                "why_required": gap.why_required,
                "refines_gap_id": gap.refines_gap_id,
                "requested_gap_id": requested_gap_id,
                "protocol_notes": list(protocol_notes),
                "capability_queries": list(gap.capability_queries),
                "reference_queries": list(gap.reference_queries),
                "seen_ids": set(),
                "ordered_seen_ids": [],
                "last_new_ids": [],
                "last_projected_ids": [],
                "projected_seen_ids": [],
                "retrievals": 0,
                "last_new_count": 0,
                "status": "open",
            }
            ledger[gap.gap_id] = entry
            if parent is not None:
                parent["status"] = "superseded"
                parent["superseded_by"] = gap.gap_id

            performed_retrieval = True
            found: dict[str, dict[str, object]] = {}
            ref_records: dict[str, dict[str, object]] = {}
            for query in gap.capability_queries:
                for item in self.catalog.search(query, limit=8):
                    provider_id = str(item.get("provider_id", ""))
                    if not provider_id:
                        continue
                    evidence_id = canonical_engineering_evidence_id("capability", provider_id)
                    found[evidence_id] = {
                        "kind": "capability",
                        "reference": provider_id,
                        "query": query,
                        "result": item,
                    }
                    observed_by_id[evidence_id] = ObservedEngineeringEvidence(
                        evidence_id=evidence_id,
                        kind="capability",
                        reference=provider_id,
                        summary=(
                            f"Capability provider {provider_id}: {item.get('name', '')} "
                            f"({item.get('provider_type', '')}, OpenFOAM {item.get('openfoam_version', '')})"
                        )[:1200],
                    )

            for query in gap.reference_queries:
                for item in self.references.search(query, scope=gap.reference_scope, limit=6):
                    reference = str(item.get("reference", ""))
                    if not reference:
                        continue
                    evidence_id = canonical_engineering_evidence_id("openfoam_reference", reference)
                    record = {
                        "kind": "openfoam_reference",
                        "reference": reference,
                        "query": query,
                        "snippet": str(item.get("snippet", ""))[:700],
                    }
                    found[evidence_id] = record
                    ref_records.setdefault(reference, record)

            read_budget = gap.read_top_reference_matches
            for reference, record in list(ref_records.items())[:read_budget]:
                try:
                    excerpt = self.references.read(reference, start_line=1, line_count=48)[:4000]
                except (OSError, ValueError):
                    excerpt = ""
                if excerpt:
                    record["content_excerpt"] = excerpt
                    record["target_case_files"] = list(gap.target_case_files)

            for evidence_id, record in found.items():
                if record["kind"] != "openfoam_reference":
                    continue
                reference = str(record["reference"] )
                detail = str(record.get("content_excerpt") or record.get("snippet") or "")
                observed_by_id[evidence_id] = ObservedEngineeringEvidence(
                    evidence_id=evidence_id,
                    kind="openfoam_reference",
                    reference=reference,
                    summary=(f"Installed OpenFOAM reference {reference}: {detail[:1000]}")[:1200],
                )

            seen_ids = entry.setdefault("seen_ids", set())
            if not isinstance(seen_ids, set):
                seen_ids = set(seen_ids)
                entry["seen_ids"] = seen_ids
            new_ids = [evidence_id for evidence_id in found if evidence_id not in seen_ids]
            seen_ids.update(found)
            ordered_seen = entry.setdefault("ordered_seen_ids", [])
            if not isinstance(ordered_seen, list):
                ordered_seen = list(ordered_seen)
                entry["ordered_seen_ids"] = ordered_seen
            for evidence_id in found:
                if evidence_id not in ordered_seen:
                    ordered_seen.append(evidence_id)
            entry["last_new_ids"] = list(new_ids)
            for evidence_id in new_ids:
                new_candidate_records[evidence_id] = found[evidence_id]
            entry["retrievals"] = int(entry.get("retrievals", 0)) + 1
            entry["last_new_count"] = len(new_ids)
            entry["status"] = "evidence_available" if new_ids else "stagnant"
            result = {
                "gap_id": gap.gap_id,
                "status": "new_evidence" if new_ids else "no_new_evidence",
                "new_evidence_ids": new_ids,
                "new_evidence": [found[eid] for eid in new_ids],
                "total_seen": len(seen_ids),
            }
            if requested_gap_id != gap.gap_id:
                result["requested_gap_id"] = requested_gap_id
            if protocol_notes:
                result["protocol_notes"] = protocol_notes
            gap_results.append(result)

        if performed_retrieval:
            self._retrieval_cycles[phase] = cycles + 1
        new_total = sum(len(item.get("new_evidence_ids", [])) for item in gap_results)
        stagnant = [
            item["gap_id"]
            for item in gap_results
            if item.get("status") in {
                "no_new_evidence",
                "already_retrieved_blocked",
                "unknown_refinement_parent",
                "satisfied_parent_blocked",
            }
        ]
        payload = {
            "cycle": self._retrieval_cycles.get(phase, cycles),
            "cycle_limit": limit,
            "gaps": gap_results,
        }
        promoted_ids = self._select_new_evidence_for_model(gap_results, new_candidate_records)
        promoted_set = set(promoted_ids)
        for item in gap_results:
            projected = [eid for eid in item.get("new_evidence_ids", []) if eid in promoted_set]
            item["projected_evidence_ids"] = projected
            ledger_entry = ledger.get(str(item.get("gap_id", "")))
            if isinstance(ledger_entry, dict):
                ledger_entry["last_projected_ids"] = list(projected)
                prior = list(ledger_entry.get("projected_seen_ids", []) or [])
                for evidence_id in projected:
                    if evidence_id not in prior:
                        prior.append(evidence_id)
                ledger_entry["projected_seen_ids"] = prior
        observed = [observed_by_id[eid] for eid in promoted_ids if eid in observed_by_id]
        durable_observed = [
            observed_by_id[eid]
            for eid in new_candidate_records
            if eid in observed_by_id
        ]
        payload["retrieved_new_evidence_count"] = new_total
        payload["projected_new_evidence_count"] = len(observed)
        payload["projection_limit"] = self.policy.max_new_model_evidence_per_batch
        payload_ref = self._store_evidence_payload(
            state,
            phase=phase,
            step=step,
            action_type=action.type,
            payload=payload,
            observed_evidence=durable_observed,
        )
        return self._event(
            step,
            action.type,
            True,
            (
                f"Evidence-gap batch completed: {new_total} retrieved; "
                f"{len(observed)} promoted to model context (limit={self.policy.max_new_model_evidence_per_batch}); "
                f"stagnant={len(stagnant)}."
            ),
            self._evidence_batch_display(
                gap_results,
                cycle=self._retrieval_cycles.get(phase, cycles),
                limit=limit,
            ),
            payload_ref=payload_ref,
            observed_evidence=observed,
        )

    def _runtime_case_file_contract_scan(self, state: CFDState) -> dict[str, object]:
        """Scan solve-critical text files for systematic FoamFile contract failures.

        This is intentionally deterministic and batch-oriented. A runtime failure in one
        IOobject must not force the LLM to discover the same missing-header defect one file
        per foamRun attempt. Core system files, every current initial field, and every
        Agent-declared required solve input are scanned together.
        """

        seals = {item.path: item for item in self.workspace.file_seals()}
        candidates: list[str] = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
        if state.engineering_plan is not None:
            candidates.extend(state.engineering_plan.required_case_files)
        candidates.extend(path for path in seals if path.startswith("0/"))
        candidates.extend(path for path in seals if path.startswith("system/"))

        invalid: list[dict[str, object]] = []
        checked: list[dict[str, object]] = []
        seen: set[str] = set()
        for path in candidates:
            if path in seen or path not in seals:
                continue
            seen.add(path)
            suffix = Path(path).suffix.lower()
            if suffix in {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".emesh"}:
                continue
            try:
                text = self.workspace.read_text(path)
            except (OSError, WorkspaceSafetyError):
                continue
            result = validate_foam_file_header(
                path,
                text,
                expected_class=("dictionary" if path.startswith("system/") else None),
            )
            record = {
                "path": path,
                "class": result.header.class_name or None,
                "object": result.header.object_name or None,
                "valid": result.valid,
            }
            checked.append(record)
            if not result.valid:
                invalid.append({**record, "failures": list(result.failures)})

        return {
            "checked_count": len(checked),
            "invalid_count": len(invalid),
            "invalid": invalid[:20],
        }

    def _runtime_relevant_case_files(self, state: CFDState, runtime_log: str | None) -> list[dict[str, object]]:
        """Return a bounded, failure-focused case slice for runtime repair."""
        seals = {item.path: item for item in self.workspace.file_seals()}
        candidates: list[str] = ["system/fvSchemes", "system/fvSolution", "system/controlDict"]
        text = runtime_log or ""
        for match in re.findall(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", text):
            probe = match.rstrip("./")
            parts = probe.split("/")
            while parts:
                candidate = "/".join(parts)
                if candidate in seals:
                    candidates.append(candidate)
                    break
                parts.pop()
        required = state.engineering_plan.required_case_files if state.engineering_plan is not None else []
        for path in required:
            base = path.rsplit("/", 1)[-1]
            if base and re.search(rf"(?<![A-Za-z0-9_]){re.escape(base)}(?![A-Za-z0-9_])", text):
                candidates.append(path)
        result: list[dict[str, object]] = []
        seen: set[str] = set()
        for path in candidates:
            if path in seen or path not in seals:
                continue
            seen.add(path)
            try:
                content = self.workspace.read_text(path)
            except (OSError, WorkspaceSafetyError):
                continue
            result.append(
                {
                    "path": path,
                    "sha256": seals[path].sha256,
                    "content": content[:8000],
                    "truncated": len(content) > 8000,
                }
            )
            if len(result) >= 6:
                break
        return result

    def _validation_relevant_case_files(
        self,
        state: CFDState,
        plan: EngineeringPlan | None,
        diagnostic: str,
        *,
        max_files: int = 8,
    ) -> list[dict[str, object]]:
        """Return exact current files implicated by one committed-case validation failure.

        This is deliberately narrower than ``current_case_files``.  The live failure
        that motivated v4.7.4 named only ``fvSchemes`` but the old repair path resent
        the entire Engineering state and exceeded the 18k context budget before the
        Agent could fix the file.  Basename matching is required because path
        redaction intentionally turns an absolute ``.../system/fvSchemes`` path into
        ``<LOCAL_PATH:fvSchemes>`` in model-facing diagnostics.
        """
        seals = {item.path: item for item in self.workspace.file_seals()}
        text = diagnostic or ""
        candidates: list[str] = []

        # Prefer literal case-relative paths that survived diagnostic redaction.
        for match in re.findall(r"(?:0|constant|system)/[A-Za-z0-9_.\/-]+", text):
            probe = match.rstrip("./")
            parts = probe.split("/")
            while parts:
                candidate = "/".join(parts)
                if candidate in seals:
                    candidates.append(candidate)
                    break
                parts.pop()

        # Redacted native diagnostics often preserve only the basename.  Match it
        # against currently sealed files rather than guessing a directory.
        for path in seals:
            base = path.rsplit("/", 1)[-1]
            if base and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(base)}(?![A-Za-z0-9_])",
                text,
            ):
                candidates.append(path)

        # If the diagnostic names a confirmed implementation binding, include that
        # artifact even when the native text did not print a path.
        if plan is not None:
            for binding in plan.confirmed_fact_bindings:
                for path in binding.case_files:
                    base = path.rsplit("/", 1)[-1]
                    if path in seals and base and re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(base)}(?![A-Za-z0-9_])",
                        text,
                    ):
                        candidates.append(path)

        # Core dictionaries are small, useful neighbours for otherwise path-poor
        # pre-solve diagnostics.  They come after explicit matches, so the failed
        # file always receives the first content budget.
        for core in ("system/fvSchemes", "system/fvSolution", "system/controlDict"):
            if core in seals:
                candidates.append(core)

        result: list[dict[str, object]] = []
        seen: set[str] = set()
        for path in candidates:
            if path in seen or path not in seals:
                continue
            seen.add(path)
            try:
                content = self.workspace.read_text(path)
            except (OSError, WorkspaceSafetyError):
                continue
            result.append(
                {
                    "path": path,
                    "sha256": seals[path].sha256,
                    "content": content,
                    "truncated": False,
                }
            )
            if len(result) >= max_files:
                break
        return result

    def _dispatch_tool_action(self, action, *, step, native_execution, phase, state=None):
        if state is not None:
            state.pending_action = {"status": "intent", "phase": phase, "step": step,
                                    "action": action.model_dump(mode="json")}
            self.checkpoint(state, "action-intent")
        event = self._dispatch_tool_action_impl(action, step=step, native_execution=native_execution, phase=phase, state=state)
        if state is not None:
            state.pending_action = {"status": "completed", "event": event.model_dump(mode="json")}
            add_support_observation(state, event)
            self.checkpoint(state, "action-completed")
            state.pending_action = None
        return event

    def _dispatch_tool_action_impl(
        self,
        action,
        *,
        step: int,
        native_execution: bool,
        phase: str,
        state: CFDState | None = None,
    ) -> EngineeringEvent:
        try:
            # Enforce effects BEFORE syntax preflight can itself create processes.
            # The LLM's role label has no authority over this decision.
            if isinstance(action, RunNativeOpenFOAMAction):
                deny_unapproved_engineering_solve(action.invocation.command, action.invocation.arguments)
            if isinstance(action, RunMeshCommandAction):
                deny_unapproved_engineering_solve(action.command, [])
            if phase.startswith("runtime") and state is not None and isinstance(
                action, (WriteCaseFileAction, PatchCaseFileAction, DeleteCaseFileAction)
            ):
                from pathlib import PurePosixPath
                path = PurePosixPath(action.patch.path if isinstance(action, PatchCaseFileAction) else action.path)
                if state.execution_approval is None:
                    raise ExecutionPolicyError("Runtime repair lacks a durable execution approval.")
                if not (path.parts[0] == "system" and path.name in {"fvSchemes", "fvSolution"}):
                    self._repair_review_required = "This repair changes more than numerical settings; explicit user review is required."
                    raise ExecutionPolicyError(self._repair_review_required)
            if phase.startswith("runtime") and isinstance(action, (RunNativeOpenFOAMAction, RunMeshCommandAction)):
                command = action.invocation.command if isinstance(action, RunNativeOpenFOAMAction) else action.command
                arguments = action.invocation.arguments if isinstance(action, RunNativeOpenFOAMAction) else []
                if command_effect(command, arguments) in {"mesh", "initialization", "write", "decomposition"}:
                    self._repair_review_required = "Native repair changes physical/mesh inputs; explicit user review is required."
                    raise ExecutionPolicyError(self._repair_review_required)
            if (
                native_execution
                and state is not None
                and isinstance(
                    action,
                    (
                        ValidateDictionaryAction,
                        SurfaceCheckAction,
                        RunMeshCommandAction,
                        RunNativeOpenFOAMAction,
                        ValidatePreSolveAction,
                    ),
                )
                and self._native_command_count(state) >= self.policy.max_native_commands
            ):
                return self._event(
                    step,
                    getattr(action, "type", "unknown"),
                    False,
                    f"Native OpenFOAM command budget exhausted ({self.policy.max_native_commands}); no command was executed.",
                    validation_status="fail", failure_category="infra",
                )

            if isinstance(action, GatherEvidenceAction):
                if state is None:
                    return self._event(
                        step, action.type, False,
                        "Evidence retrieval requires an active CFDState for structured payload storage.",
                        failure_signature=f"evidence_retrieval:{phase}:missing_state",
                        failure_scope="pipeline",
                    )
                return self._gather_evidence(action, step=step, phase=phase, state=state)

            if isinstance(action, InspectEnvironmentAction):
                payload = {
                    "runtime": self.tools.environment_snapshot(),
                    "capability_graph": self.catalog.summary(),
                    "reference_roots": self.references.summary(),
                }
                return self._event(step, action.type, True, "Environment inspected.", _json(payload))

            if isinstance(action, SearchCapabilitiesAction):
                results = self.catalog.search(action.query)
                observed = [
                    ObservedEngineeringEvidence(
                        evidence_id=canonical_engineering_evidence_id(
                            "capability", str(item["provider_id"])
                        ),
                        kind="capability",
                        reference=str(item["provider_id"]),
                        summary=(
                            f"Capability provider {item['provider_id']}: {item.get('name', '')} "
                            f"({item.get('provider_type', '')}, OpenFOAM {item.get('openfoam_version', '')})"
                        )[:1200],
                    )
                    for item in results
                    if isinstance(item, dict) and item.get("provider_id")
                ]
                payload_ref = (
                    self._store_evidence_payload(
                        state, phase=phase, step=step, action_type=action.type,
                        payload=results, observed_evidence=observed,
                    )
                    if state is not None else None
                )
                display = "\n".join(
                    f"{item.evidence_id}: {item.summary}" for item in observed[:12]
                )
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Capability search returned {len(results)} provider(s).",
                    display,
                    payload_ref=payload_ref,
                    observed_evidence=observed,
                )

            if isinstance(action, SearchReferencesAction):
                results = self.references.search(action.query, scope=action.scope)
                observed = [
                    ObservedEngineeringEvidence(
                        evidence_id=canonical_engineering_evidence_id(
                            "openfoam_reference", str(item["reference"])
                        ),
                        kind="openfoam_reference",
                        reference=str(item["reference"]),
                        summary=(
                            f"Installed OpenFOAM reference {item['reference']}: "
                            f"{str(item.get('snippet', ''))[:700]}"
                        )[:1200],
                    )
                    for item in results
                    if isinstance(item, dict) and item.get("reference")
                ]
                payload_ref = (
                    self._store_evidence_payload(
                        state, phase=phase, step=step, action_type=action.type,
                        payload=results, observed_evidence=observed,
                    )
                    if state is not None else None
                )
                display = "\n".join(
                    f"{item.evidence_id}: {item.summary}" for item in observed[:12]
                )
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Reference search returned {len(results)} result(s).",
                    display,
                    payload_ref=payload_ref,
                    observed_evidence=observed,
                )

            if isinstance(action, ReadReferenceAction):
                text = self.references.read(
                    action.reference,
                    start_line=action.start_line,
                    line_count=action.line_count,
                )
                observed = [
                    ObservedEngineeringEvidence(
                        evidence_id=canonical_engineering_evidence_id(
                            "openfoam_reference", action.reference
                        ),
                        kind="openfoam_reference",
                        reference=action.reference,
                        summary=f"Read installed OpenFOAM reference {action.reference}.",
                    )
                ]
                payload_ref = (
                    self._store_evidence_payload(
                        state, phase=phase, step=step, action_type=action.type,
                        payload={"reference": action.reference, "content": text, "target_case_files": action.target_case_files, "read_metadata": getattr(self.references, "last_read_metadata", {})},
                        observed_evidence=observed,
                    )
                    if state is not None else None
                )
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Read {action.reference}.",
                    compact_text(text, min(self.policy.model_event_excerpt_chars, 2500)),
                    payload_ref=payload_ref,
                    observed_evidence=observed,
                )

            if isinstance(action, ListCaseFilesAction):
                files = [
                    {
                        "path": item.path,
                        "sha256": item.sha256,
                        "size_bytes": item.size_bytes,
                    }
                    for item in self.workspace.file_seals()
                ]
                return self._event(step, action.type, True, "Listed agent-authored case files.", _json(files))

            if isinstance(action, ReadCaseFileAction):
                text = self.workspace.read_text(action.path)
                return self._event(step, action.type, True, f"Read {action.path}.", text)

            # v4.2.1: do not require a reference excerpt merely to mutate an
            # OpenFOAM case file. Workspace policy rejects executable directives,
            # unsafe includes/libraries/paths and oversized content; downstream
            # dictionary/native validation establishes implementation correctness.

            if isinstance(action, WriteCaseFileAction):
                if action.path.startswith("postprocessConfig/"):
                    return self._event(
                        step,
                        action.type,
                        False,
                        "postprocessConfig/ is reserved for the post-processing phase after a successful solve.",
                    )
                if state is not None:
                    cycles, failure_pending, current_cycle_started = self._mesh_repair_status(state)
                    if (
                        failure_pending
                        and not current_cycle_started
                        and cycles >= self.policy.max_mesh_repair_cycles
                    ):
                        return self._event(
                            step,
                            action.type,
                            False,
                            f"Mesh repair cycle budget exhausted ({self.policy.max_mesh_repair_cycles}); case edit was not applied.",
                        )
                mesh_affecting = self.workspace.is_mesh_affecting_path(action.path)
                normalized_path = action.path
                expected_digest = hashlib.sha256(action.content.encode("utf-8")).hexdigest()
                if self._precommitted_files.get(normalized_path) == expected_digest:
                    digest = expected_digest
                    self._precommitted_files.pop(normalized_path, None)
                else:
                    digest = self.workspace.write_text(action.path, action.content)
                self._invalidate_mesh_dependencies(state)
                if action.path == "system/blockMeshDict":
                    # Generic text writes have no trustworthy structured topology representation.
                    # Higher-level block_mesh actions restore this registry after the write succeeds.
                    self._structured_block_mesh = None
                self._presolve_case_manifest = None
                self._presolve_required_case_files = None
                if mesh_affecting:
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                if phase == "runtime_repair" and state is not None:
                    # Any runtime repair mutation invalidates the user-approved case seal
                    # until retry_solver revalidates and reseals the current case.
                    state.case_seal = None
                return self._event(
                    step,
                    action.type,
                    True,
                    f"Wrote {action.path}.",
                    artifact_sha256=digest,
                )

            if isinstance(action, PatchCaseFileAction):
                patch = action.patch
                if state is not None:
                    cycles, failure_pending, current_cycle_started = self._mesh_repair_status(state)
                    if failure_pending and not current_cycle_started and cycles >= self.policy.max_mesh_repair_cycles:
                        return self._event(
                            step,
                            action.type,
                            False,
                            f"Mesh repair cycle budget exhausted ({self.policy.max_mesh_repair_cycles}); case patch was not applied.",
                        )
                mesh_affecting = self.workspace.is_mesh_affecting_path(patch.path)
                digest = self.workspace.patch_text_once(patch.path, patch.old, patch.new)
                self._invalidate_mesh_dependencies(state)
                if patch.path == "system/blockMeshDict":
                    self._structured_block_mesh = None
                self._presolve_case_manifest = None
                self._presolve_required_case_files = None
                if mesh_affecting:
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                if phase == "runtime_repair" and state is not None:
                    state.case_seal = None
                return self._event(
                    step, action.type, True, f"Patched {patch.path}.", artifact_sha256=digest
                )

            if isinstance(action, DeleteCaseFileAction):
                if state is not None:
                    cycles, failure_pending, current_cycle_started = self._mesh_repair_status(state)
                    if (
                        failure_pending
                        and not current_cycle_started
                        and cycles >= self.policy.max_mesh_repair_cycles
                    ):
                        return self._event(
                            step,
                            action.type,
                            False,
                            f"Mesh repair cycle budget exhausted ({self.policy.max_mesh_repair_cycles}); case delete was not applied.",
                        )
                mesh_affecting = self.workspace.is_mesh_affecting_path(action.path)
                if action.path in self._precommitted_drops:
                    self._precommitted_drops.discard(action.path)
                else:
                    self.workspace.delete(action.path)
                self._invalidate_mesh_dependencies(state)
                if action.path == "system/blockMeshDict":
                    self._structured_block_mesh = None
                self._presolve_case_manifest = None
                self._presolve_required_case_files = None
                if mesh_affecting:
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                if phase == "runtime_repair" and state is not None:
                    state.case_seal = None
                return self._event(step, action.type, True, f"Deleted {action.path}.")

            if isinstance(action, ValidateDictionaryAction):
                target = self.workspace.resolve_case_path(action.path, must_exist=True)
                text = target.read_text(encoding="utf-8", errors="replace")
                header = validate_foam_file_header(
                    action.path, text,
                    expected_class=("dictionary" if action.path.startswith("system/") else None),
                )
                if not header.valid:
                    return self._event(
                        step, action.type, False,
                        f"OpenFOAM file header rejected {action.path}.", header.render(),
                        validation_status="fail", failure_category="case",
                    )
                if not native_execution or not self.policy.foam_dictionary_probe:
                    return self._event(
                        step, action.type, True,
                        f"Static FoamFile/header validation accepted {action.path}; optional foamDictionary probe skipped.",
                        header.render(), validation_status="pass",
                    )
                result = self.tools.foam_dictionary_validate(target, cwd=self.workspace.case_dir)
                output = _tool_output(result)
                self.workspace.write_log(f"{step:03d}.foamDictionary.log", output)
                assessment = classify_native_validation(result, command_name="foamDictionary", probe=True)
                event_output = assessment.diagnostic.render() if assessment.diagnostic is not None else output
                if assessment.status == "pass":
                    summary = f"Optional foamDictionary probe accepted {action.path}."
                elif assessment.status == "inconclusive":
                    summary = f"Optional foamDictionary probe was inconclusive for {action.path}; continuing to stronger consumer validation."
                else:
                    summary = f"foamDictionary reported an explicit case-level fatal diagnostic for {action.path}."
                return self._event(
                    step, action.type, assessment.workflow_success, summary, event_output,
                    native_command_executed=True, validation_status=assessment.status,
                    failure_category=assessment.category,
                )

            if isinstance(action, SurfaceCheckAction):
                if not native_execution:
                    return self._event(
                        step, action.type, True,
                        "Native surfaceCheck unavailable; geometry validation remains inconclusive until a consumer uses the surface.",
                        validation_status="inconclusive", failure_category="tool",
                    )
                target = self.workspace.resolve_case_path(action.path, must_exist=True)
                result = self.tools.surface_check(target, cwd=self.workspace.case_dir)
                output = _tool_output(result)
                self.workspace.write_log(f"{step:03d}.surfaceCheck.log", output)
                assessment = classify_native_validation(result, command_name="surfaceCheck", probe=True)
                event_output = assessment.diagnostic.render() if assessment.diagnostic is not None else output
                summary = (
                    f"surfaceCheck passed for {action.path}." if assessment.status == "pass"
                    else (f"surfaceCheck was inconclusive for {action.path}; continuing to downstream consumer validation."
                          if assessment.status == "inconclusive"
                          else f"surfaceCheck reported a case-level fatal diagnostic for {action.path}.")
                )
                return self._event(
                    step, action.type, assessment.workflow_success, summary, event_output,
                    native_command_executed=True, validation_status=assessment.status,
                    failure_category=assessment.category,
                )

            if isinstance(action, RunNativeOpenFOAMAction):
                invocation = action.invocation
                if not native_execution:
                    return self._event(
                        step, action.type, False,
                        f"Native execution is disabled; {invocation.command} was not run.",
                        validation_status="inconclusive", failure_category="infra",
                    )
                preflight = self.safety.validate_native_inputs()
                if not preflight.valid:
                    return self._event(
                        step, action.type, False,
                        f"{invocation.command} blocked by deterministic static preflight.",
                        "\n".join(preflight.failures), validation_status="fail", failure_category="case",
                    )
                if invocation.command == "snappyHexMesh":
                    precondition_ok, precondition_reason = self._mesh_command_precondition(invocation.command)
                    if not precondition_ok:
                        return self._event(
                            step, "mesh_tool_precondition", False,
                            f"{invocation.command} blocked by deterministic executable precondition.",
                            precondition_reason, failure_signature=f"tool_contract:{invocation.command}:prerequisite",
                            failure_scope="strategy", validation_status="fail", failure_category="case",
                        )
                try:
                    result = self.tools.run_native_command(
                        invocation.command, self.workspace.case_dir, arguments=invocation.arguments
                    )
                except (ValueError, WorkspaceSafetyError) as exc:
                    return self._event(
                        step, action.type, False, str(exc),
                        validation_status="fail", failure_category="security",
                    )
                if invocation.command != "checkMesh":
                    self._presolve_case_manifest = None
                    self._presolve_required_case_files = None
                effect = command_effect(invocation.command, invocation.arguments)
                if effect in {"mesh", "decomposition", "write", "initialization"}:
                    from openfoam_agent.contracts.mesh_dependencies import MeshDependencyGraph
                    MeshDependencyGraph(self.workspace).record_native(
                        invocation.command, invocation.arguments,
                        self._pending_execution_plan or (state.engineering_plan if state else None),
                    )
                    self._invalidate_mesh_dependencies(state)
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                        state.case_seal = None
                        state.solve_approved = False
                        state.execution_approval = None
                output = _tool_output(result)
                self.workspace.write_log(f"{step:03d}.{invocation.command}.log", output)
                assessment = classify_native_validation(
                    result, command_name=invocation.command,
                    probe=(effect == "validation" and invocation.command != "checkMesh"),
                )
                event_output = assessment.diagnostic.render() if assessment.diagnostic is not None else output
                failure_signature = None
                failure_scope = None
                if assessment.diagnostic is not None:
                    failure_signature = self._native_failure_signature(
                        invocation.command, assessment.diagnostic.kind, assessment.diagnostic.excerpt
                    )
                    failure_scope = "local"
                event_success = assessment.workflow_success
                validation_status = assessment.status
                failure_category = assessment.category
                summary = assessment.reason
                if assessment.status == "inconclusive" and effect != "validation":
                    event_success = False
                if invocation.command == "checkMesh" and state is not None:
                    if assessment.status == "pass":
                        evidence = parse_check_mesh_evidence(result)
                        self._record_region_mesh(state, evidence, invocation.arguments)
                        event_success = evidence.passed
                        validation_status = "pass" if evidence.passed else "fail"
                        failure_category = None if evidence.passed else "case"
                        summary = f"checkMesh returned status {result.return_code}; evidence {'passed' if evidence.passed else 'failed'}."
                        if evidence.passed:
                            self._checkmesh_mesh_manifest = self.workspace.mesh_manifest_digest()
                    else:
                        event_success = False
                        summary = ("checkMesh validation was inconclusive; mesh validity was not invented."
                                   if assessment.status == "inconclusive" else assessment.reason)
                return self._event(
                    step, action.type, event_success, summary, event_output,
                    native_command_executed=True,
                    mesh_command_executed=(effect in {"mesh", "validation", "decomposition"}),
                    failure_signature=failure_signature, failure_scope=failure_scope,
                    validation_status=validation_status, failure_category=failure_category,
                )

            if isinstance(action, ValidatePreSolveAction):
                if not native_execution:
                    return self._event(
                        step, action.type, True,
                        "Native consumer validation is disabled; static pre-solve validation is deferred.",
                        validation_status="inconclusive", failure_category="infra",
                    )
                result = self.presolve.validate_required_case_files(action.required_case_files)
                if not result.valid:
                    return self._event(
                        step, action.type, False, "Pre-solve deterministic readiness validation failed.",
                        "\n".join(result.failures), validation_status="fail", failure_category="case",
                    )
                output_lines = [f"checkedFiles={len(result.checked_files)}", f"meshPatches={len(result.mesh_patches)}"]
                output_lines.extend(f"warning: {item}" for item in result.warnings[:12])
                validation_status = "pass"
                failure_category = None
                native_ran = False
                plan = self._pending_execution_plan or (state.engineering_plan if state is not None else None)
                validator = getattr(self.tools, "zero_step_consumer_validate", None)
                if (self.policy.zero_step_consumer_validation and plan is not None and plan.execution is not None and callable(validator)):
                    native_result, note = validator(
                        self.workspace.case_dir, plan.execution, timeout=self.policy.zero_step_validation_timeout
                    )
                    output_lines.append(note)
                    if native_result is None:
                        validation_status = "inconclusive"
                        failure_category = "tool"
                    else:
                        native_ran = True
                        native_output = _tool_output(native_result)
                        self.workspace.write_log(f"{step:03d}.zeroStepConsumer.log", native_output)
                        assessment = classify_native_validation(native_result, command_name=plan.execution.driver, probe=False)
                        output_lines.append(assessment.diagnostic.render() if assessment.diagnostic is not None else native_output)
                        if assessment.status == "fail":
                            return self._event(
                                step, action.type, False, "Zero-step OpenFOAM consumer initialization rejected the case.",
                                "\n".join(output_lines), native_command_executed=True,
                                validation_status="fail", failure_category="case",
                            )
                        if assessment.status == "inconclusive":
                            validation_status = "inconclusive"
                            failure_category = assessment.category or "tool"
                self._presolve_case_manifest = self.workspace.manifest_digest()
                self._presolve_required_case_files = tuple(action.required_case_files)
                return self._event(
                    step, action.type, True,
                    ("Pre-solve readiness passed; zero-step consumer validation was inconclusive but did not prove the case invalid."
                     if validation_status == "inconclusive" else "Pre-solve readiness and consumer initialization validation passed."),
                    "\n".join(output_lines), native_command_executed=native_ran,
                    validation_status=validation_status, failure_category=failure_category,
                )

            if isinstance(action, RunMeshCommandAction):
                if not native_execution:
                    return self._event(
                        step, action.type, False, f"Native execution is disabled; {action.command} was not run.",
                        validation_status="inconclusive", failure_category="infra",
                    )
                preflight = self.safety.validate_native_inputs()
                if not preflight.valid:
                    return self._event(
                        step, action.type, False, f"{action.command} blocked by deterministic static preflight.",
                        "\n".join(preflight.failures), validation_status="fail", failure_category="case",
                    )
                precondition_ok, precondition_reason = self._mesh_command_precondition(action.command)
                if not precondition_ok:
                    signature = f"tool_contract:{action.command}:prerequisite"
                    return self._event(
                        step, "mesh_tool_precondition", False,
                        f"{action.command} blocked by deterministic executable precondition.", precondition_reason,
                        failure_signature=signature, failure_scope="strategy",
                        validation_status="fail", failure_category="case",
                    )
                result = self.tools.run_mesh_command(action.command, self.workspace.case_dir)
                effect = command_effect(action.command, [])
                if effect in {"mesh", "decomposition", "initialization", "write"}:
                    from openfoam_agent.contracts.mesh_dependencies import MeshDependencyGraph
                    MeshDependencyGraph(self.workspace).record_native(
                        action.command, [], self._pending_execution_plan or (state.engineering_plan if state else None)
                    )
                    self._invalidate_mesh_dependencies(state)
                if action.command != "checkMesh":
                    self._presolve_case_manifest = None
                    self._presolve_required_case_files = None
                if action.command in _MESH_TOPOLOGY_MUTATING_COMMANDS:
                    self._checkmesh_mesh_manifest = None
                    if state is not None:
                        state.mesh_evidence = None
                        state.case_seal = None
                        state.solve_approved = False
                        state.execution_approval = None
                output = _tool_output(result)
                self.workspace.write_log(f"{step:03d}.{action.command}.log", output)
                assessment = classify_native_validation(result, command_name=action.command, probe=False)
                event_output = assessment.diagnostic.render() if assessment.diagnostic is not None else output
                failure_signature = None
                failure_scope = None
                if assessment.diagnostic is not None:
                    failure_signature = self._native_failure_signature(
                        action.command, assessment.diagnostic.kind, assessment.diagnostic.excerpt
                    )
                    failure_scope = "local"
                event_success = assessment.status == "pass"
                validation_status = assessment.status
                failure_category = assessment.category
                summary = assessment.reason
                if action.command == "checkMesh" and state is not None and assessment.status == "pass":
                    evidence = parse_check_mesh_evidence(result)
                    self._record_region_mesh(state, evidence, [])
                    event_success = evidence.passed
                    validation_status = "pass" if evidence.passed else "fail"
                    failure_category = None if evidence.passed else "case"
                    summary = f"checkMesh returned status {result.return_code}; evidence {'passed' if evidence.passed else 'failed'}."
                    if evidence.passed:
                        self._checkmesh_mesh_manifest = self.workspace.mesh_manifest_digest()
                elif assessment.status == "inconclusive":
                    event_success = False
                    summary = f"{action.command} was inconclusive; native mesh state was not guessed."
                return self._event(
                    step, action.type, event_success, summary, event_output,
                    native_command_executed=True, mesh_command_executed=True,
                    failure_signature=failure_signature, failure_scope=failure_scope,
                    validation_status=validation_status, failure_category=failure_category,
                )
        except (ValueError, FileNotFoundError, WorkspaceSafetyError, OSError, ExecutionPolicyError) as exc:
            if isinstance(action, GatherEvidenceAction):
                reason = f"{type(exc).__name__}: {exc}"
                self._disable_evidence_retrieval(phase, reason)
                diagnostic = reason[:280]
                return self._event(
                    step, action.type, False,
                    "Evidence retrieval infrastructure failed once; further retrieval is disabled for this phase. "
                    f"Cause: {diagnostic}. Proceed with existing evidence/authorized engineering defaults or block.",
                    reason, failure_signature=f"evidence_retrieval:{phase}:infrastructure",
                    failure_scope="pipeline", validation_status="fail", failure_category="infra",
                )
            if isinstance(exc, (WorkspaceSafetyError, ExecutionPolicyError)):
                category = "security"
            elif isinstance(exc, FileNotFoundError):
                category = "case"
            elif isinstance(exc, OSError):
                category = "infra"
            else:
                category = "case"
            return self._event(
                step, getattr(action, "type", "unknown"), False,
                f"Tool action rejected: {type(exc).__name__}: {exc}",
                validation_status="fail", failure_category=category,
            )

        return self._event(
            step, getattr(action, "type", "unknown"), False, f"Action is not valid in {phase} phase.",
            validation_status="fail", failure_category="infra",
        )

    @staticmethod
    def _revision_record(
        state: CFDState,
        *,
        proposal_id: str,
        feedback_ids: list[str],
        before_plan: EngineeringPlan,
        before_seal,
        after_plan: EngineeringPlan,
        after_seal,
    ) -> RevisionRecord:
        before = {item.path: item.sha256 for item in before_seal.files}
        after = {item.path: item.sha256 for item in after_seal.files}
        changes: list[RevisionFileChange] = []
        for path in sorted(before.keys() | after.keys()):
            old = before.get(path)
            new = after.get(path)
            if old == new:
                continue
            if old is None:
                change = "added"
            elif new is None:
                change = "removed"
            else:
                change = "modified"
            changes.append(
                RevisionFileChange(
                    path=path,
                    change=change,
                    before_sha256=old,
                    after_sha256=new,
                )
            )
        return RevisionRecord(
            revision_id=f"rev-{len(state.revision_history) + 1:04d}",
            proposal_id=proposal_id,
            feedback_ids=list(feedback_ids),
            before_plan_sha256=before_plan.digest(),
            after_plan_sha256=after_plan.digest(),
            before_manifest_sha256=before_seal.manifest_sha256,
            after_manifest_sha256=after_seal.manifest_sha256,
            archive_path=state.pending_revision_archive_path,
            file_changes=changes,
        )

    def _case_plan_authoring_failure_count(self, state: CFDState) -> int:
        return sum(
            1
            for event in self._current_round_events(state)
            if (
                not event.success
                and event.action_type in {"typed_dictionary_serialize", "block_mesh_serialize", "case_bundle_preflight", "authoring_semantic_conflict", "case_build_graph"}
            )
        )

    def _record_case_plan_authoring_failure(
        self,
        state: CFDState,
        event: EngineeringEvent,
    ) -> bool:
        """Append a pre-commit failure and stop after a small dedicated retry budget.

        Returns True when the workflow was transitioned to ENGINEERING_BLOCKED.
        """

        state.engineering_events.append(event)
        failures = self._case_plan_authoring_failure_count(state)
        if failures >= self.policy.max_case_plan_authoring_retries:
            state.transition(
                State.ENGINEERING_BLOCKED,
                "Complete case-plan authoring repeatedly failed deterministic pre-commit checks "
                f"({failures}/{self.policy.max_case_plan_authoring_retries}).",
            )
            return True
        return False

    def _candidate_block_mesh_retry_required(self, state: CFDState) -> bool:
        events = self._current_round_events(state)
        if not events:
            return False
        last = events[-1]
        return (
            not last.success
            and last.action_type == "block_mesh_serialize"
            and self._pending_execution_plan is None
            and self._pending_candidate_execution is not None
            and self._pending_candidate_execution.block_mesh is not None
        )

    def _case_plan_retry_required(self, state: CFDState) -> bool:
        """Return whether the previous complete-plan authoring attempt failed pre-commit.

        These failures leave the workspace intentionally untouched, but the rejected
        complete candidate remains in Python memory. The compact retry contract therefore
        accepts only a delta against that retained candidate (or block).
        """

        events = self._current_round_events(state)
        if not events:
            return False
        last = events[-1]
        return (
            not last.success
            and last.action_type in {"typed_dictionary_serialize", "case_bundle_preflight", "authoring_semantic_conflict", "case_build_graph"}
            and self._pending_execution_plan is None
            and self._pending_candidate_execution is not None
        )

    def _mesh_tool_contracts(self) -> list[dict[str, object]]:
        provider = getattr(self.tools, "mesh_tool_contracts", None)
        if callable(provider):
            value = provider()
            return list(value) if isinstance(value, list) else []
        return []

    def _mesh_command_precondition(self, command: str) -> tuple[bool, str]:
        checker = getattr(self.tools, "mesh_command_precondition", None)
        if callable(checker):
            return checker(command, self.workspace.case_dir)
        return True, ""

    def _native_toolchain_preflight(
        self, commands: list[str] | tuple[str, ...]
    ) -> tuple[bool, tuple[str, ...], int]:
        """Check selected native executables under the actual sanitized loader env.

        This is controller-owned infrastructure validation. It never chooses a CFD
        strategy and never turns a loader/package problem into a case repair. Tool
        doubles/offline tests that do not expose the capability remain compatible.
        """
        checker = getattr(self.tools, "native_command_preflight", None)
        if not callable(checker):
            return True, (), 0
        failures: list[str] = []
        checked = 0
        for command in dict.fromkeys(str(item) for item in commands if str(item)):
            try:
                status = checker(command)
            except (OSError, ValueError, RuntimeError, ExecutionPolicyError) as exc:
                failures.append(f"{command}: native toolchain preflight failed: {type(exc).__name__}: {exc}")
                continue
            checked += 1
            available = bool(status.get("available", True))
            trusted = bool(status.get("trusted", True))
            ready = bool(status.get("runtime_ready", available and trusted))
            if available and trusted and ready:
                continue
            reason = str(status.get("reason") or "native executable/toolchain is unavailable")
            missing = status.get("missing_libraries")
            if isinstance(missing, list) and missing:
                reason += " | missingLibraries=" + ",".join(str(item) for item in missing[:12])
            failures.append(f"{command}: {reason}")
        return not failures, tuple(failures), checked

    @staticmethod
    def _native_failure_signature(command: str, kind: str, excerpt: str) -> str:
        """Normalize a native failure enough to detect repeated no-progress repairs."""
        primary = ""
        for line in excerpt.splitlines():
            text = line.strip()
            if not text:
                continue
            lowered = text.casefold()
            if (
                "foam fatal" in lowered
                or lowered.startswith("file:")
                or lowered.startswith("from function")
                or lowered.startswith("in file")
                or text.startswith("#")
            ):
                continue
            primary = text
            break
        if not primary:
            primary = kind
        primary = re.sub(r"<[^>]+>", "<PATH>", primary)
        # blockMesh diagnostics commonly embed transient cell/face/vertex labels in an
        # otherwise identical topology error. Those labels must not defeat repeated-
        # failure detection (e.g. face 4(3 4 20 19) vs 4(3 19 20 4)). Preserve the
        # message structure while normalizing only numeric labels for blockMesh.
        if command == "blockMesh":
            primary = re.sub(
                r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                "<N>",
                primary,
            )
        primary = re.sub(r"\s+", " ", primary).strip().casefold()
        digest = hashlib.sha256(f"{command}\0{kind}\0{primary}".encode("utf-8", errors="replace")).hexdigest()[:20]
        return f"native:{command}:{kind}:{digest}"

    def _strategy_revision_required(self, state: CFDState) -> bool:
        events = self._current_round_events(state)
        failure_indexes = [index for index, event in enumerate(events) if not event.success]
        if not failure_indexes:
            return False
        # Escalation applies only to the *current unresolved* failure. A successful
        # strategy revision after that failure resolves the trigger and allows the
        # controller to return to authoring/validation. A later fresh failure will
        # create a new trigger.
        last_index = failure_indexes[-1]
        last = events[last_index]
        if any(
            event.success and event.action_type == "revise_mesh_strategy"
            for event in events[last_index + 1 :]
        ):
            return False
        if not last.failure_signature:
            return False
        if last.failure_scope == "strategy":
            return True
        same = [
            event for event in events[: last_index + 1]
            if (not event.success) and event.failure_signature == last.failure_signature
        ]
        return len(same) >= 2 and last.mesh_command_executed

    def _block_mesh_repair_required(self, state: CFDState) -> bool:
        if self._pending_execution_plan is None or self._structured_block_mesh is None:
            return False
        events = self._current_round_events(state)
        for event in reversed(events):
            if event.mesh_command_executed:
                return bool(
                    not event.success
                    and event.failure_signature
                    and str(event.failure_signature).startswith("native:blockMesh:")
                )
            if not event.success and event.action_type == "block_mesh_repair_preflight":
                return True
        return False

    def _phase_contract(self, state: CFDState, phase: str):
        from openfoam_agent.engineering.phases.context_controller import phase_contract
        return phase_contract(self, state, phase)

    def _engineering_conversation_key(self, state: CFDState, contract_phase: str) -> str:
        from openfoam_agent.engineering.phases.context_controller import engineering_conversation_key
        return engineering_conversation_key(self, state, contract_phase)

    def _geometry_authoring_policy(self, state: CFDState) -> dict[str, object]:
        """Controller-owned geometry ownership/fidelity policy for progress-first CFD design.

        Missing external CAD is not automatically missing user input. When no immutable
        user asset is present and the confirmed intake describes geometry conceptually,
        the Engineering Agent owns the representative procedural geometry and may create
        safe case-local text geometry (for example blockMesh topology or ASCII STL).
        """
        assets = list(state.assets or [])
        geometry_facts = []
        if state.intake is not None:
            for fact in state.intake.facts:
                if str(fact.id).startswith("geometry."):
                    geometry_facts.append({
                        "id": fact.id,
                        "value": fact.value,
                        "source": fact.source,
                    })
        return {
            "mode": "progress_first_geometry_v1",
            "user_assets_present": bool(assets),
            "user_assets": assets[:16],
            "confirmed_geometry_facts": geometry_facts[:24],
            "agent_generated_geometry_authorized": True,
            "representative_geometry_defaults_authorized": True,
            "preferred_self_contained_methods": [
                "typed blockMesh for simple channels/pipes/obstacles",
                "agent-authored ASCII STL/OBJ plus native meshing when a surface method is genuinely useful",
            ],
            "rules": [
                "A missing external CAD/STL is not a blocking condition unless an exact user-owned asset is itself a confirmed requirement.",
                "Creating representative procedural geometry from confirmed topology plus engineering_defaults is authorized engineering, not fabrication.",
                "Do not overwrite or silently replace immutable user assets.",
                "If a chosen surface-based mesh requires geometry and no user asset is required, author the case-local surface or choose an equivalent self-contained procedural mesh instead of blocking.",
                "Do not require a large Agent-generated triangulated surface when it cannot be completely represented within the bounded authoring contract; use a compact procedural strategy or allow controller-routed strategy revision.",
                "Record representative dimensions/angles/radii selected by the Agent as engineering_defaults and do not claim exact geometric fidelity.",
            ],
        }

    @staticmethod
    def _agent_owned_generated_geometry_paths(state: CFDState, plan: EngineeringPlan | None) -> tuple[str, ...]:
        """Return solve-required geometry artifacts that are not immutable user assets."""
        if plan is None:
            return ()
        asset_paths = {
            str(item.get("case_path") or "")
            for item in (state.assets or [])
            if isinstance(item, dict) and item.get("case_path")
        }
        generated: list[str] = []
        for path in plan.required_case_files:
            lowered = str(path).casefold()
            is_surface = (
                lowered.startswith("constant/trisurface/")
                or lowered.startswith("constant/geometry/")
                or lowered.endswith((".stl", ".obj", ".off", ".ply"))
            )
            if is_surface and path not in asset_paths:
                generated.append(path)
        return tuple(dict.fromkeys(generated))

    def _recoverable_precommit_authoring_block(
        self, state: CFDState, action: BlockAction
    ) -> tuple[bool, tuple[str, ...]]:
        """Recognize an Agent-owned authoring strategy that should be replanned, not terminally blocked.

        The controller does not choose replacement geometry. It only distinguishes a
        pre-commit implementation-feasibility failure from missing user input, then
        routes the next model turn to strategy revision.
        """
        plan = self._draft_design_plan
        if plan is None or action.needs_user_input:
            return False, ()
        generated = self._agent_owned_generated_geometry_paths(state, plan)
        if action.block_kind == "authoring_strategy_infeasible":
            return True, generated
        if action.block_kind not in {"other", "engineering_choice_missing"}:
            return False, generated
        if not generated:
            return False, generated
        diagnostic = " ".join([action.reason, *action.missing_items]).casefold()
        markers = (
            "surface", "stl", "obj", "geometry", "trisurface", "cad",
            "artifact", "incomplete", "cannot author", "unable to author",
            "too large", "output limit", "bounded",
        )
        return any(marker in diagnostic for marker in markers), generated

    def _revision_target_file_records(self, state: CFDState) -> list[dict[str, object]]:
        seals = {item.path: item for item in self.workspace.file_seals()}
        result: list[dict[str, object]] = []
        for path in state.revision_target_case_files[:12]:
            if path not in seals:
                continue
            try:
                content = self.workspace.read_text(path)
            except (OSError, WorkspaceSafetyError):
                continue
            result.append({
                "path": path,
                "sha256": seals[path].sha256,
                "content": content[:7000],
                "truncated": len(content) > 7000,
            })
        return result

    def _execute_revision_decision(self, state: CFDState, action: RevisionDecisionAction, *, step: int) -> bool:
        if state.engineering_plan is None:
            raise WorkspaceSafetyError("Revision decision requires the sealed baseline EngineeringPlan.")
        baseline = state.engineering_plan
        if action.plan_patch is not None:
            revised = action.plan_patch.apply(baseline)
            if revised.confirmed_intake_sha256 != state.intake_digest:
                raise WorkspaceSafetyError("Revision plan_patch changed the frozen confirmed intake identity.")
            failures = validate_design(revised, state.intake)
            failures.extend(self._validate_engineering_defaults(revised, state))
            if failures:
                event = self._event(step, action.type, False, "Human revision plan decision rejected.", "\n".join(failures), failure_category="case")
                state.engineering_events.append(event)
                return False
            state.pending_revision_plan = revised
        else:
            state.pending_revision_plan = baseline
        targets = list(action.target_case_files)
        if not targets and action.plan_patch is None:
            event = self._event(step, action.type, False, "Revision decision contained neither a plan change nor target case files.")
            state.engineering_events.append(event)
            return False
        state.revision_target_case_files = targets
        state.revision_decision_complete = True
        event = self._event(
            step, action.type, True,
            "Human revision engineering decision accepted; case-delta authoring is now isolated to the next phase.",
            f"targets={len(targets)}; planChanged={action.plan_patch is not None}",
        )
        state.engineering_events.append(event)
        return False

    def _generate_turn(self, state: CFDState, *, step: int, local_step: int | None = None, current_step_limit: int | None = None, phase: str, runtime_log: str | None = None, native_execution: bool = True):
        from openfoam_agent.engineering.phases.context_controller import generate_turn
        return generate_turn(self, state, step=step, local_step=local_step, current_step_limit=current_step_limit, phase=phase, runtime_log=runtime_log, native_execution=native_execution)

    def _validate_engineering_defaults(
        self,
        plan: EngineeringPlan,
        state: CFDState,
    ) -> list[str]:
        failures: list[str] = []
        failures.extend(f"EngineeringPlan semantic conflict: {item}" for item in getattr(plan, "plan_conflicts", []))
        # v4.2 progress-first policy: ordinary engineering defaults are allowed whenever
        # they do not override confirmed user facts. Provenance remains explicit and
        # downstream validation/review decides whether the chosen value was adequate.
        registry = self._observed_evidence_registry(state)
        for default in plan.engineering_defaults:
            for evidence_id in default.evidence_ids:
                if evidence_id not in registry:
                    failures.append(
                        f"Engineering default {default.parameter!r} references unobserved evidence ID {evidence_id}."
                    )
        return failures

    def _validate_observed_provenance(
        self,
        plan: EngineeringPlan,
        state: CFDState,
    ) -> list[str]:
        """Reject execution/capability claims that Python did not issue in this run."""

        failures: list[str] = []
        registry = self._observed_evidence_registry(state)
        capability_ids = {
            item.reference: item.evidence_id
            for item in registry.values()
            if item.kind == "capability"
        }
        requirements: list[tuple[str, str, set[str], str]] = []
        execution = plan.execution
        if execution is None:
            requirements.append((
                plan.solver_provider_id, plan.solver, {"solver", "solver_module", "generated_solver"}, "solver"
            ))
        else:
            requirements.append((
                execution.driver_provider_id, execution.driver,
                {"execution_driver", "solver_application", "utility"}, "execution driver"
            ))
            if execution.driver == "foamRun":
                assert execution.solver_module is not None and execution.solver_provider_id is not None
                requirements.append((
                    execution.solver_provider_id, execution.solver_module,
                    {"solver", "solver_module", "generated_solver"}, "solver module"
                ))
            elif execution.driver == "foamMultiRun":
                for item in execution.regions:
                    requirements.append((
                        item.provider_id, item.solver_module,
                        {"solver", "solver_module", "generated_solver"},
                        f"region solver {item.region}",
                    ))

        seen_requirements: set[str] = set()
        for provider_id, expected_name, allowed_types, label in requirements:
            if provider_id in seen_requirements:
                continue
            seen_requirements.add(provider_id)
            provider = self.catalog.provider(provider_id)
            if provider is None:
                failures.append(f"{label.title()} provider '{provider_id}' does not exist in the capability catalog.")
                continue
            if provider.provider_type not in allowed_types:
                failures.append(
                    f"Capability provider '{provider_id}' has type {provider.provider_type!r}, "
                    f"which is not valid for {label}."
                )
            if provider.name != expected_name:
                failures.append(
                    f"{label.title()} '{expected_name}' disagrees with capability provider "
                    f"'{provider_id}' ({provider.name})."
                )
            if provider.openfoam_version != plan.openfoam_version:
                failures.append(
                    f"Capability provider '{provider_id}' targets OpenFOAM "
                    f"{provider.openfoam_version}, not {plan.openfoam_version}."
                )
            if provider_id not in capability_ids:
                executable = provider.provider_type in {"execution_driver", "solver_application", "utility"}
                if not provider_is_sufficient(provider, executable=executable):
                    failures.append(
                        f"Capability provider '{provider_id}' lacks sufficient deterministic installed "
                        "evidence for this design stage."
                    )

        # Opaque evidence IDs are integrity pointers: if the Agent explicitly claims one,
        # it must have been issued by the deterministic registry. The Agent may simply
        # omit advisory evidence instead of fabricating a pointer.
        for evidence in plan.evidence:
            if evidence.evidence_id not in registry:
                failures.append(
                    "Engineering evidence ID was not issued by the deterministic evidence "
                    f"registry in this run: {evidence.evidence_id}"
                )
        return failures

    def _evidence_details_for_model(self, state: CFDState) -> dict[str, object]:
        """Build bounded evidence details from the structured evidence store.

        The durable payload never rides inside EngineeringEvent.output_excerpt.  This
        projection exposes only the evidence item relevant to each canonical ID.
        """

        details: dict[str, object] = {}
        for record in state.engineering_evidence_records:
            payload = record.payload
            if record.action_type == "gather_evidence" and isinstance(payload, dict):
                for gap in payload.get("gaps", []) or []:
                    if not isinstance(gap, dict):
                        continue
                    ids = list(gap.get("new_evidence_ids", []) or [])
                    items = list(gap.get("new_evidence", []) or [])
                    for evidence_id, item in zip(ids, items):
                        if isinstance(evidence_id, str):
                            details[evidence_id] = item
            elif record.action_type == "search_capabilities" and isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict) or not item.get("provider_id"):
                        continue
                    evidence_id = canonical_engineering_evidence_id(
                        "capability", str(item["provider_id"])
                    )
                    details[evidence_id] = item
            elif record.action_type == "search_references" and isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict) or not item.get("reference"):
                        continue
                    evidence_id = canonical_engineering_evidence_id(
                        "openfoam_reference", str(item["reference"])
                    )
                    details[evidence_id] = item
            elif record.action_type == "read_reference" and isinstance(payload, dict):
                reference = str(payload.get("reference", ""))
                if reference:
                    evidence_id = canonical_engineering_evidence_id(
                        "openfoam_reference", reference
                    )
                    details[evidence_id] = {
                        "reference": reference,
                        "content_excerpt": compact_text(str(payload.get("content", "")), 2200),
                    }
        return details

    def _available_evidence_for_model(self, state: CFDState) -> list[dict[str, object]]:
        registry = self._observed_evidence_registry(state)
        details = self._evidence_details_for_model(state)
        records: list[dict[str, object]] = []
        for evidence in registry.values():
            item = evidence.model_dump(mode="json")
            detail = details.get(evidence.evidence_id)
            if detail is not None:
                # Bound each structured detail independently; the model still gets the
                # canonical evidence descriptor even when verbose payload is compacted.
                encoded = json.dumps(detail, ensure_ascii=False, separators=(",", ":"), default=str)
                item["detail"] = json.loads(compact_text(encoded, 2600)) if len(encoded) <= 2600 else compact_text(encoded, 2600)
            records.append(item)
        return records

    def _bounded_evidence_for_model(
        self,
        state: CFDState,
        *,
        phase: str,
        max_items: int,
    ) -> list[dict[str, object]]:
        """Compile a relevance/recency-bounded evidence capsule for one LLM turn.

        Durable evidence remains complete in CFDState.  The model sees only evidence
        associated with the active/recent gaps plus a small fallback tail.  This
        prevents successful retrieval from making every later prompt monotonically
        larger.
        """

        if max_items <= 0:
            return []
        registry = self._observed_evidence_registry(state)
        details = self._evidence_details_for_model(state)
        selected: list[str] = []
        selected_set: set[str] = set()

        def add(evidence_id: object) -> None:
            if not isinstance(evidence_id, str):
                return
            if evidence_id not in registry or evidence_id in selected_set:
                return
            selected.append(evidence_id)
            selected_set.add(evidence_id)

        ledger_items = list(self._evidence_gap_ledger.get(phase, {}).items())
        active = [
            item for item in ledger_items
            if str(item[1].get("status", "")) not in {"superseded"}
        ]
        # Keep evidence from at most the four most recent active gaps.  Per-gap caps
        # preserve breadth when one broad query returns dozens of weak matches.
        for _gap_id, entry in reversed(active[-4:]):
            last_new = list(entry.get("last_projected_ids", []) or [])
            projected_seen = list(entry.get("projected_seen_ids", []) or [])
            # v4.0.3: only evidence promoted by the retrieval projection policy is
            # eligible for the next bounded model capsule. The full retrieved set
            # remains in the durable evidence store for traceability.
            for evidence_id in last_new[:4]:
                add(evidence_id)
            for evidence_id in projected_seen[:6]:
                add(evidence_id)

        # Track retrieval candidates that were deliberately *not* promoted. They remain
        # available in the durable ledger, but must not leak back into this model turn via
        # targeted-capability or generic registry fallbacks.
        unprojected_retrieval_ids: set[str] = set()
        for record in state.engineering_evidence_records:
            if record.action_type != "gather_evidence" or not isinstance(record.payload, dict):
                continue
            for gap in record.payload.get("gaps", []) or []:
                if not isinstance(gap, dict):
                    continue
                retrieved = {str(eid) for eid in gap.get("new_evidence_ids", []) or []}
                projected_ids = {str(eid) for eid in gap.get("projected_evidence_ids", []) or []}
                unprojected_retrieval_ids.update(retrieved - projected_ids)

        # Before generic fallback, surface intake-relevant providers that Python
        # deterministically pre-observed from the frozen intake, unless this retrieval
        # batch deliberately left that provider unprojected.
        for provider_id in self._targeted_capability_provider_ids(state):
            evidence_id = canonical_engineering_evidence_id("capability", provider_id)
            if evidence_id not in unprojected_retrieval_ids:
                add(evidence_id)

        # If the gap ledger is sparse, preserve the most recent deterministic records.
        for record in reversed(state.engineering_evidence_records[-6:]):
            if record.action_type == "gather_evidence" and isinstance(record.payload, dict):
                projected_ids: list[str] = []
                for gap in record.payload.get("gaps", []) or []:
                    if isinstance(gap, dict):
                        projected_ids.extend(str(eid) for eid in gap.get("projected_evidence_ids", []) or [])
                for evidence_id in projected_ids[:6]:
                    add(evidence_id)
            else:
                for observed in record.observed_evidence[:6]:
                    add(observed.evidence_id)

        # Finally retain a tiny provider/reference tail so a no-gap design still has
        # deterministic capability anchors.
        for evidence in registry.values():
            if evidence.evidence_id in unprojected_retrieval_ids:
                continue
            if evidence.kind == "capability":
                add(evidence.evidence_id)
            if len(selected) >= max_items:
                break
        if len(selected) < max_items:
            for evidence_id in reversed(list(registry)):
                if evidence_id in unprojected_retrieval_ids:
                    continue
                add(evidence_id)
                if len(selected) >= max_items:
                    break

        projected: list[dict[str, object]] = []
        for evidence_id in selected[:max_items]:
            evidence = registry[evidence_id]
            item: dict[str, object] = {
                "evidence_id": evidence.evidence_id,
                "kind": evidence.kind,
                "reference": compact_text(evidence.reference, 320),
                "summary": compact_text(evidence.summary, 520),
            }
            detail = details.get(evidence_id)
            if detail is not None:
                encoded = json.dumps(
                    detail, ensure_ascii=False, separators=(",", ":"), default=str
                )
                item["detail"] = compact_text(
                    encoded, self.policy.max_model_evidence_detail_chars
                )
            projected.append(item)
        return projected

    def _compact_evidence_gap_status(
        self, phase: str, *, max_gaps: int = 6
    ) -> list[dict[str, object]]:
        status = self._evidence_gap_status(phase)
        if len(status) <= max_gaps:
            return status
        return status[-max_gaps:]

    def _intake_capability_queries(self, state: CFDState) -> list[str]:
        """Derive a tiny deterministic capability-search shortlist from frozen intake facts.

        This does not choose a solver/model.  It only pre-observes likely relevant
        documented/installed providers so the Agent can often decide without an extra
        gather_evidence round trip.
        """

        if state.intake is None:
            return []
        preferred = ("classification", "physics", "objective", "material", "motion")
        queries: list[str] = []
        for prefix in preferred:
            for fact in state.intake.facts:
                if fact.category != prefix:
                    continue
                text = str(fact.value).replace("_", " ").strip()
                if len(text) < 3 or text in queries:
                    continue
                queries.append(text[:160])
                if len(queries) >= 5:
                    return queries
        return queries

    def _targeted_capability_provider_ids(self, state: CFDState) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for query in self._intake_capability_queries(state):
            for item in self.catalog.search(query, limit=4):
                provider_id = str(item.get("provider_id", ""))
                if not provider_id or provider_id in seen:
                    continue
                seen.add(provider_id)
                ids.append(provider_id)
                if len(ids) >= 12:
                    return ids
        return ids

    def _observed_evidence_registry(
        self,
        state: CFDState,
    ) -> dict[str, ObservedEngineeringEvidence]:
        registry: dict[str, ObservedEngineeringEvidence] = {}

        # v2.9 fast path: a small capability graph can be deterministically exposed
        # before the first LLM call. This is still Python-issued evidence; the model
        # chooses the solver/provider, while Python later validates the choice.
        if self.policy.preload_capabilities:
            for item in self.catalog.search("", limit=self.policy.max_preloaded_capabilities):
                provider_id = str(item.get("provider_id", ""))
                if not provider_id:
                    continue
                evidence = ObservedEngineeringEvidence(
                    evidence_id=canonical_engineering_evidence_id("capability", provider_id),
                    kind="capability",
                    reference=provider_id,
                    summary=(
                        f"Preloaded capability provider {provider_id}: {item.get('name', '')} "
                        f"({item.get('provider_type', '')}, OpenFOAM {item.get('openfoam_version', '')})"
                    )[:1200],
                )
                registry[evidence.evidence_id] = evidence

            for provider_id in self._targeted_capability_provider_ids(state):
                provider = self.catalog.provider(provider_id)
                if provider is None:
                    continue
                evidence = ObservedEngineeringEvidence(
                    evidence_id=canonical_engineering_evidence_id("capability", provider_id),
                    kind="capability",
                    reference=provider_id,
                    summary=(
                        f"Targeted intake capability provider {provider_id}: {provider.name} "
                        f"({provider.provider_type}, OpenFOAM {provider.openfoam_version})"
                    )[:1200],
                )
                registry[evidence.evidence_id] = evidence

        for record in state.engineering_evidence_records:
            for item in record.observed_evidence:
                registry[item.evidence_id] = item

        # Backward compatibility for states created before the structured evidence store.
        for event in state.engineering_events:
            if not event.success:
                continue
            for item in event.observed_evidence:
                registry[item.evidence_id] = item
        return dict(sorted(registry.items()))

    def _cumulative_provenance_summary(self, state: CFDState) -> dict[str, object]:
        successful = [event for event in state.engineering_events if event.success]
        registry = self._observed_evidence_registry(state)
        provider_ids = sorted(
            item.reference for item in registry.values() if item.kind == "capability"
        )
        reference_hints = sorted(
            self._redact_local_paths(item.summary)[:300]
            for item in registry.values()
            if item.kind == "openfoam_reference"
        )
        return {
            "successful_action_types": sorted({event.action_type for event in successful}),
            "observed_capability_provider_ids": provider_ids,
            "reference_observation_summaries": reference_hints[-12:],
            "canonical_evidence_ids": list(registry)[-40:],
            "evidence_record_refs": [item.record_id for item in state.engineering_evidence_records[-12:]],
            "mesh_evidence_passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
        }

    def _redact_event_for_model(self, event: EngineeringEvent) -> dict[str, object]:
        payload = compact_event_for_model(
            event,
            excerpt_chars=self.policy.model_event_excerpt_chars,
        )
        payload["summary"] = self._redact_local_paths(str(payload.get("summary", "")))
        payload["output_excerpt"] = self._redact_local_paths(
            str(payload.get("output_excerpt", ""))
        )
        return payload

    def _recent_observations_for_model(self, state: CFDState) -> list[dict[str, object]]:
        """Return compact LLM observations, collapsing sequence members into one record.

        Raw EngineeringEvent objects remain in CFDState for deterministic evidence,
        budgets, diagnostics and audit. Only the model projection is compacted.
        """

        groups: list[dict[str, object]] = []
        events = state.engineering_events
        index = 0
        while index < len(events):
            event = events[index]
            if not event.sequence_id:
                groups.append(self._redact_event_for_model(event))
                index += 1
                continue

            sequence_id = event.sequence_id
            members: list[EngineeringEvent] = []
            while index < len(events) and events[index].sequence_id == sequence_id:
                members.append(events[index])
                index += 1

            failed = next((item for item in members if not item.success), None)
            sequence_projection: dict[str, object] = {
                "kind": "engineering_sequence_summary",
                "step": members[0].step,
                "sequence_id": sequence_id,
                "goal": self._redact_local_paths(members[0].sequence_goal or "")[:1000],
                "success": failed is None,
                "planned_actions": members[0].sequence_length,
                "executed_actions": len(members),
                "stopped_early": bool(
                    members[0].sequence_length
                    and len(members) < members[0].sequence_length
                ),
                "actions": [
                    {
                        "action_type": item.action_type,
                        "success": item.success,
                        "summary": self._redact_local_paths(item.summary)[:500],
                        **(
                            {"artifact_sha256": item.artifact_sha256}
                            if item.artifact_sha256
                            else {}
                        ),
                    }
                    for item in members
                ],
                "native_commands_executed": sum(
                    1 for item in members if item.native_command_executed
                ),
                "mesh_commands_executed": sum(
                    1 for item in members if item.mesh_command_executed
                ),
            }
            if failed is not None and failed.output_excerpt.strip():
                sequence_projection["failure_output_excerpt"] = self._redact_local_paths(
                    failed.output_excerpt[-self.policy.model_event_excerpt_chars :]
                )
            groups.append(sequence_projection)

        return groups[-self.policy.observation_history :]

    def redact_native_observation(self, text: str) -> str:
        """Return a display/model-safe native diagnostic without changing its meaning."""
        return self._redact_local_paths(text)

    def _redact_local_paths(self, text: str) -> str:
        replacements: list[tuple[str, str]] = []
        known = [
            (str(self.workspace.case_dir), "<CASE_DIR>"),
            (str(self.workspace.root), "<WORKSPACE>"),
            (os.environ.get("WM_PROJECT_DIR", ""), "<OPENFOAM_ROOT>"),
            (os.environ.get("HOME", ""), "<HOME>"),
        ]
        for raw, marker in known:
            if raw:
                replacements.append((str(Path(raw).expanduser()), marker))
                try:
                    replacements.append((str(Path(raw).expanduser().resolve()), marker))
                except OSError:
                    pass
        redacted = text
        for raw, marker in sorted(set(replacements), key=lambda item: len(item[0]), reverse=True):
            redacted = redacted.replace(raw, marker)

        # Remove any other absolute Unix path appearing in model-bound tool/log text.
        # Preserve only the basename as a diagnostic hint. User-provided prompt text is
        # not passed through this function.
        absolute_path = re.compile(r"(?<![A-Za-z0-9_:/])/(?P<body>[A-Za-z0-9._~+-][^\s\"'<>|;()]*)")

        def replace_path(match: re.Match[str]) -> str:
            raw = "/" + match.group("body")
            core = raw.rstrip(".,:")
            suffix = raw[len(core):]
            name = Path(core).name
            marker = f"<LOCAL_PATH:{name}>" if name else "<LOCAL_PATH>"
            return marker + suffix

        return absolute_path.sub(replace_path, redacted)

    def _emit_action_started(
        self,
        phase: str,
        action: object,
        *,
        step: int,
        limit: int,
    ) -> None:
        importance = action_importance(str(getattr(action, "type", "action")))
        self.progress.emit(
            ProgressEvent(
                phase=phase,
                message=describe_action(action),
                status="start",
                step=step,
                limit=limit,
                importance=importance,
            )
        )

    def _emit_engineering_event(
        self,
        phase: str,
        event: EngineeringEvent,
        *,
        step: int,
        limit: int,
        state: CFDState | None = None,
    ) -> None:
        metrics: dict[str, object] = {}
        if event.action_type == "run_mesh_command" and state is not None and state.mesh_evidence is not None:
            evidence = state.mesh_evidence
            metrics = {
                "cells": evidence.cell_count,
                "maxNonOrtho": evidence.max_non_orthogonality,
                "maxSkew": evidence.max_skewness,
            }
        details: tuple[str, ...] = ()
        if event.output_excerpt.strip() and (not event.success or event.action_type == "finish_preview"):
            if event.action_type == "finish_preview":
                details = tuple(
                    self._redact_local_paths(line.strip())[:800]
                    for line in event.output_excerpt.splitlines()
                    if line.strip()
                )[:12]
            elif event.native_command_executed:
                details = tuple(
                    self._redact_local_paths(line.rstrip())[:800]
                    for line in event.output_excerpt.splitlines()
                    if line.strip()
                )[:ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT]
        progress_status = (
            "info" if event.validation_status == "inconclusive"
            else ("success" if event.success else "failure")
        )
        self.progress.emit(
            ProgressEvent(
                phase=phase,
                message=event.summary,
                status=progress_status,
                step=step,
                limit=limit,
                importance=action_importance(event.action_type),
                metrics=metrics,
                details=details,
            )
        )

    def _event(
        self,
        step: int,
        action_type: str,
        success: bool,
        summary: str,
        output: str = "",
        *,
        payload_ref: str | None = None,
        artifact_sha256: str | None = None,
        native_command_executed: bool = False,
        mesh_command_executed: bool = False,
        failure_signature: str | None = None,
        failure_scope: str | None = None,
        validation_status: str | None = None,
        failure_category: str | None = None,
        observed_evidence: list[ObservedEngineeringEvidence] | None = None,
    ) -> EngineeringEvent:
        # EngineeringEvent is a bounded progress/audit projection, never the durable
        # storage location for large tool payloads. compact_text accounts for its own
        # marker, avoiding the old 12000 + truncation-marker overflow.
        output_limit = min(self.policy.max_observation_chars, 12_000)
        summary = compact_text(str(summary), 4000)
        output = compact_text(str(output), output_limit) if output else ""
        # Events are a bounded projection only.  The durable EngineeringEvidenceRecord
        # retains the complete observed-evidence set referenced by payload_ref.  Never let
        # a large but valid retrieval batch turn a progress/UI cardinality limit into a
        # workflow-fatal validation error.
        projected_evidence = sorted(
            list(observed_evidence or []),
            key=lambda item: (item.kind, item.reference, item.evidence_id),
        )[:ENGINEERING_EVENT_OBSERVED_EVIDENCE_LIMIT]
        return EngineeringEvent(
            step=step,
            action_type=action_type,
            success=success,
            summary=summary,
            output_excerpt=output,
            payload_ref=payload_ref,
            artifact_sha256=artifact_sha256,
            native_command_executed=native_command_executed,
            mesh_command_executed=mesh_command_executed,
            failure_signature=failure_signature,
            failure_scope=failure_scope,
            validation_status=(validation_status or ("pass" if success else "fail")),
            failure_category=failure_category,
            observed_evidence=projected_evidence,
        )


def _tool_output(result) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
