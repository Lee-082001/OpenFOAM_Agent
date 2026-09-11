from __future__ import annotations

from openfoam_agent.contracts.evidence import implementation_evidence_pack, evidence_coverage_failures, authoring_prompt_evidence, advisory_authoring_evidence_summary
from openfoam_agent.contracts.evidence_policy import POLICY_SUMMARY, provider_is_sufficient
from openfoam_agent.engineering.authoring_tasks import compile_tasks, accept_task
from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.engineering.case_delta_graph import compile_case_delta_graph
from openfoam_agent.llm.context import ContextBudgetError
from openfoam_agent.llm.context_capsules import project_confirmed_intake, project_plan_core, project_feedback_history
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

# v4.8 physical phase boundary: model-context/schema routing lives outside the facade.


def phase_contract(self, state: CFDState, phase: str):
    if not self.policy.compact_phase_schemas:
        return EngineeringTurn, ENGINEERING_SYSTEM_PROMPT, "legacy"
    if phase in {"prepare_finalize", "human_revision_finalize"}:
        return FinalizationTurn, FINALIZATION_SYSTEM_PROMPT, "finalize"
    if phase == "runtime_repair":
        return RuntimeRepairTurn, RUNTIME_REPAIR_SYSTEM_PROMPT, "runtime_repair"
    if phase.startswith("human_revision"):
        if not state.revision_decision_complete:
            return RevisionDecisionTurn, REVISION_DECISION_SYSTEM_PROMPT, "revision_decide"
        return RevisionAuthoringTurn, REVISION_AUTHORING_SYSTEM_PROMPT, "revision_author"
    if phase == "prepare" and self._candidate_block_mesh_retry_required(state):
        return (
            CandidateBlockMeshRepairTurn,
            CANDIDATE_BLOCK_MESH_REPAIR_SYSTEM_PROMPT,
            "block_mesh_replan",
        )
    if phase == "prepare" and self._case_plan_retry_required(state):
        return CasePlanRetryTurn, CASE_PLAN_RETRY_SYSTEM_PROMPT, "replan"
    if phase == "prepare" and self._strategy_revision_required(state):
        return StrategyRevisionTurn, STRATEGY_REVISION_SYSTEM_PROMPT, "strategy_revision"
    if phase == "prepare" and self._block_mesh_repair_required(state):
        return BlockMeshRepairTurn, BLOCK_MESH_REPAIR_SYSTEM_PROMPT, "block_mesh_repair"
    if phase == "prepare" and self._pending_execution_plan is not None:
        return RepairTurn, REPAIR_SYSTEM_PROMPT, "repair"
    if phase == "prepare" and self.policy.staged_case_authoring:
        if self._draft_design_plan is not None:
            return CaseAuthoringTurn, CASE_AUTHORING_SYSTEM_PROMPT, "author_case"
        if self._retrieval_cycles.get("prepare", 0) >= self.policy.max_prepare_retrieval_cycles:
            return (
                PrepareDecisionDesignTurn,
                PREPARE_DECISION_DESIGN_SYSTEM_PROMPT,
                "prepare_design_decide",
            )
        return PrepareDesignTurn, PREPARE_DESIGN_SYSTEM_PROMPT, "prepare_design"
    if phase == "prepare" and self._retrieval_cycles.get("prepare", 0) >= self.policy.max_prepare_retrieval_cycles:
        return PrepareDecisionOnlyTurn, PREPARE_DECISION_ONLY_SYSTEM_PROMPT, "prepare_decide"
    return PrepareTurn, PREPARE_SYSTEM_PROMPT, "prepare"


def engineering_conversation_key(self, state: CFDState, contract_phase: str) -> str:
    if contract_phase == "runtime_repair":
        return f"eng:{state.run_id}:runtime:{state.simulation_attempts}"
    if contract_phase in {"revision", "revision_decide", "revision_author"}:
        proposal = state.active_revision_proposal
        suffix = proposal.proposal_id if proposal is not None else str(len(state.revision_history))
        return f"eng:{state.run_id}:revision:{suffix}"
    return f"eng:{state.run_id}:round:{state.engineering_round_start_index}"


def generate_turn(
    self,
    state: CFDState,
    *,
    step: int,
    local_step: int | None = None,
    current_step_limit: int | None = None,
    phase: str,
    runtime_log: str | None = None,
    native_execution: bool = True,
):
    if local_step is None:
        local_step = step
    if current_step_limit is None:
        current_step_limit = (
            self.policy.max_runtime_repair_steps
            if phase == "runtime_repair"
            else self.policy.max_agent_steps
        )

    schema, system_prompt, contract_phase = self._phase_contract(state, phase)
    conversation_key = self._engineering_conversation_key(state, contract_phase)
    prompt_count = self._phase_prompt_counts.get(conversation_key, 0)
    plan_digest = state.engineering_plan.digest() if state.engineering_plan is not None else None
    manifest_digest = self.workspace.manifest_digest()
    evidence_total = len(self._observed_evidence_registry(state))
    if self.policy.bounded_evidence_context:
        if contract_phase == "author_case":
            evidence_records = []
        else:
            evidence_limit = (
                self.policy.max_decide_model_evidence_items
                if contract_phase in {"prepare_design_decide", "prepare_decide"}
                else self.policy.max_prepare_model_evidence_items
            )
            evidence_records = self._bounded_evidence_for_model(
                state, phase=phase, max_items=evidence_limit
            )
    else:
        evidence_records = self._available_evidence_for_model(state)
    case_files = [
        {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
        for item in self.workspace.file_seals()
    ]
    budget = {
        "llm_limit": current_step_limit,
        "llm_remaining": max(0, current_step_limit - local_step + 1),
        "initial_engineering_step_budget": self.policy.max_agent_steps,
        "current_engineering_step_limit": current_step_limit,
        "hard_engineering_step_limit": self.policy.hard_max_agent_steps,
        "steps_remaining_in_current_window": max(0, current_step_limit - local_step + 1),
        "finalization_only": phase in {"prepare_finalize", "human_revision_finalize"},
        "tool_limit": (
            self.policy.max_runtime_repair_tool_actions
            if phase == "runtime_repair"
            else self.policy.max_tool_actions
        ),
        "tool_used": self._tool_action_count(state),
        "native_limit": self.policy.max_native_commands,
        "native_used": self._native_command_count(state),
        "retrieval_cycles_used": self._retrieval_cycles.get(phase, 0),
        "retrieval_cycle_limit": (
            self.policy.max_runtime_retrieval_cycles
            if phase == "runtime_repair"
            else self.policy.max_prepare_retrieval_cycles
        ),
    }
    bindings = {
        "intake_sha256": state.intake_digest,
        "plan_sha256": plan_digest,
        "manifest_sha256": manifest_digest,
        "check_mesh_passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
        "check_mesh_log_sha256": (
            state.mesh_evidence.raw_log_sha256 if state.mesh_evidence else None
        ),
    }
    assumption_policy = {
        "authorized": True,
        "explicit_user_authorization": bool(state.user_request.exploratory_completion_authorized),
        "policy": "progress_first_ordinary_defaults",
        "interaction_mode": state.user_request.interaction_mode,
        "provenance_for_selected_missing_values": "engineering_default",
        "allowed": [
            "representative geometry dimensions",
            "ordinary material properties",
            "inlet/initial temperatures",
            "representative flow rate or velocity",
            "representative heat-generation magnitude",
            "heated-region extent",
            "simulation duration and ordinary numerical controls",
        ],
        "forbidden": [
            "overriding a confirmed user fact",
            "claiming an engineering default was supplied by the user",
            "inventing tool/version-specific capability or syntax evidence",
        ],
    }
    evidence_policy = {
        "mode": "risk_stage_aware_v1",
        "classes": POLICY_SUMMARY,
        "hard_gate_rule": "Only mandatory evidence may block at its verification stage.",
        "deferred_rule": "Missing implementation evidence does not block design; resolve before unsafe/raw use or replace with deterministic/native validation.",
        "advisory_rule": "Missing advisory evidence never blocks by itself; record engineering provenance and validate outcomes.",
    }
    verified_execution_candidates = []
    for provider in self.catalog.all_providers():
        if provider.provider_type not in {"execution_driver", "solver_application", "solver_module"}:
            continue
        executable = provider.provider_type in {"execution_driver", "solver_application"}
        if not provider_is_sufficient(provider, executable=executable):
            continue
        verified_execution_candidates.append({
            "provider_id": provider.id,
            "name": provider.name,
            "provider_type": provider.provider_type,
            "verification_level": provider.verification_level,
            "openfoam_version": provider.openfoam_version,
            "capabilities": list(provider.capabilities)[:8],
        })
    verified_execution_candidates = verified_execution_candidates[:32]

    retrieval_policy = {
        "available": phase not in self._evidence_retrieval_disabled
        and self._retrieval_cycles.get(phase, 0) < (
            self.policy.max_runtime_retrieval_cycles
            if phase == "runtime_repair"
            else self.policy.max_prepare_retrieval_cycles
        ),
        "disabled_reason": self._evidence_retrieval_disabled.get(phase),
    }

    # Delta mode is only safe when the backend can chain the previous response.
    supports_stateful = False
    try:
        import inspect
        supports_stateful = "conversation_key" in inspect.signature(self.llm.generate).parameters
    except (TypeError, ValueError):
        supports_stateful = False
    true_stateful_delta = bool(
        supports_stateful and bool(getattr(self.llm, "store", False))
    )
    use_delta = bool(
        self.policy.state_delta_context
        and prompt_count > 0
        and (true_stateful_delta or not self.policy.bounded_evidence_context)
        and contract_phase not in {
            "runtime_repair", "block_mesh_replan", "block_mesh_repair",
            "prepare_design", "prepare_design_decide", "author_case", "revision", "revision_decide", "revision_author", "repair",
        }
    )
    previous_snapshot = self._phase_context_snapshots.get(conversation_key, {})

    if contract_phase == "author_case":
        if self._draft_design_plan is None:
            raise RuntimeError("author_case contract requested without a staged EngineeringPlan")
        payload: dict[str, object] = {
            "state_mode": "staged_case_authoring",
            "phase": phase,
            "step": step,
            "frozen_engineering_plan": project_plan_core(self._draft_design_plan, mode="authoring"),
            "frozen_engineering_plan_sha256": self._draft_design_plan.digest(),
            "confirmed_intake": project_confirmed_intake(state.intake),
            "confirmed_intake_sha256": state.intake_digest,
            "implementation_evidence_pack": advisory_authoring_evidence_summary(state, self._draft_design_plan, max_records=12),
            "assets": state.assets,
            "geometry_authoring_policy": self._geometry_authoring_policy(state),
            "authoring_brief": self._draft_authoring_brief,
            "environment_hint": self.tools.environment_snapshot(),
            "tool_execution_contracts": self._mesh_tool_contracts(),
            "current_case_files": case_files,
            "bindings": bindings,
            "budget": budget,
        }
        instruction = (
            "Author the OpenFOAM case for the frozen EngineeringPlan. Return only author_case "
            "with the required files and deterministic native validation pipeline; do not "
            "repeat/revise the plan or retrieve more evidence:\n"
        )
    elif contract_phase in {"prepare_design", "prepare_design_decide"}:
        payload = {
            "state_mode": "bounded_engineering_design",
            "phase": phase,
            "step": step,
            "confirmed_intake": project_confirmed_intake(state.intake),
            "intake_sha256": state.intake_digest,
            "engineering_assumption_policy": assumption_policy,
            "geometry_authoring_policy": self._geometry_authoring_policy(state),
            "evidence_policy": evidence_policy,
            "verified_execution_candidates": verified_execution_candidates,
            "evidence_retrieval_policy": retrieval_policy,
            "environment_hint": self.tools.environment_snapshot(),
            "capability_graph_hint": self.catalog.summary(),
            "available_evidence": evidence_records,
            "evidence_context": {
                "shown": len(evidence_records),
                "total_observed": evidence_total,
                "truncated": evidence_total > len(evidence_records),
            },
            "evidence_gap_status": self._compact_evidence_gap_status(phase),
            "recent_observations": self._recent_observations_for_model(state)[-4:],
            "bindings": bindings,
            "budget": budget,
        }
        if contract_phase == "prepare_design":
            instruction = (
                "Choose the next compact engineering-design action. Use gather_evidence only "
                "for a genuinely missing OpenFOAM tool/version fact. Otherwise return design_case "
                "with the complete EngineeringPlan but no case files. Delegated ordinary values "
                "belong in engineering_defaults:\n"
            )
        else:
            instruction = (
                "Retrieval is closed. Use this bounded evidence capsule to return design_case "
                "with the complete EngineeringPlan, or block only for a genuine unsupported "
                "tool/version requirement. Do not author case files yet:\n"
            )
    elif contract_phase == "revision_decide":
        proposal = state.active_revision_proposal
        linked_feedback = set(proposal.feedback_ids) if proposal is not None else set()
        payload = {
            "state_mode": "human_revision_decision_v2",
            "phase": phase,
            "step": step,
            "confirmed_intake": project_confirmed_intake(state.intake),
            "intake_sha256": state.intake_digest,
            "baseline_plan_sha256": plan_digest,
            "baseline_manifest_sha256": manifest_digest,
            "baseline_plan_core": project_plan_core(state.engineering_plan, mode="revision"),
            "active_revision_proposal": project_revision_proposal(proposal),
            "feedback_observations": [
                {
                    "feedback_id": item.feedback_id,
                    "scope": item.scope,
                    "statement": compact_text(item.statement, 1000),
                    "status": item.status,
                }
                for item in state.human_feedback
                if not linked_feedback or item.feedback_id in linked_feedback
            ][-4:],
            "case_manifest": [
                {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in (state.case_seal.files if state.case_seal is not None else [])
            ][:100],
            "mesh_evidence": {
                "passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
                "cell_count": state.mesh_evidence.cell_count if state.mesh_evidence else None,
                "max_non_orthogonality": state.mesh_evidence.max_non_orthogonality if state.mesh_evidence else None,
                "max_skewness": state.mesh_evidence.max_skewness if state.mesh_evidence else None,
            },
            "runtime_summary": compact_runtime_report(state.runtime_report) if state.runtime_report is not None else None,
            "budget": budget,
            "revision_contract": {
                "phase": "decision_only",
                "plan_update_interface": "plan_patch",
                "file_contents_forbidden": True,
                "next_phase": "revision_authoring",
            },
        }
        instruction = (
            "Decide the human-confirmed engineering revision before any case mutation. Return decide_revision with "
            "only the necessary plan_patch and target_case_files, or block if the confirmed proposal cannot be implemented. "
            "Do not author file contents in this turn:\n"
        )
    elif contract_phase == "revision_author":
        proposal = state.active_revision_proposal
        target_files = self._revision_target_file_records(state)
        payload = {
            "state_mode": "human_revision_authoring_v2",
            "phase": phase,
            "step": step,
            "confirmed_intake": project_confirmed_intake(state.intake),
            "intake_sha256": state.intake_digest,
            "accepted_plan_sha256": (state.pending_revision_plan.digest() if state.pending_revision_plan is not None else (state.engineering_plan.digest() if state.engineering_plan is not None else None)),
            "accepted_plan_core": project_plan_core(state.pending_revision_plan or state.engineering_plan, mode="revision"),
            "active_revision_proposal": project_revision_proposal(proposal),
            "target_case_files": list(state.revision_target_case_files),
            "target_file_records": target_files,
            "case_manifest_sha256": manifest_digest,
            "mesh_evidence": {
                "passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
                "cell_count": state.mesh_evidence.cell_count if state.mesh_evidence else None,
                "manifest_current": bool(
                    self._checkmesh_mesh_manifest and self._checkmesh_mesh_manifest == self.workspace.mesh_manifest_digest()
                ),
            },
            "supporting_observations": [
                compact_event_for_model(event, excerpt_chars=1200, summary_chars=400)
                for event in self._current_round_events(state)[-8:]
                if event.success and event.action_type in {"read_case_file", "read_reference", "search_references"}
            ][-4:],
            "budget": budget,
            "revision_contract": {
                "phase": "case_delta_authoring",
                "plan_is_already_accepted": True,
                "plan_patch_forbidden": True,
                "case_mutation_authority": "CaseDeltaGraph",
                "read_exact_files_on_demand": True,
            },
        }
        instruction = (
            "Author only the file delta required by the accepted human revision decision. Return repair_case_plan "
            "without plan_patch/updated_plan, or read a target/support file first. Do not redesign the accepted plan; "
            "CaseDeltaGraph will revalidate the complete effective case:\n"
        )
    elif contract_phase == "strategy_revision":
        baseline_plan = state.engineering_plan or self._pending_execution_plan or self._draft_design_plan
        precommit_strategy = bool(
            state.engineering_plan is None
            and self._pending_execution_plan is None
            and self._draft_design_plan is not None
        )
        recent_failures = [
            event for event in self._current_round_events(state) if not event.success
        ]
        trigger = recent_failures[-1] if recent_failures else None
        payload = {
            "state_mode": "precommit_strategy_revision_v1" if precommit_strategy else "strategy_revision_v1",
            "phase": phase,
            "step": step,
            "confirmed_facts": [
                {"id": fact.id, "value": fact.value, "source": fact.source}
                for fact in (state.intake.facts if state.intake is not None else [])
                if fact.category != "context"
            ],
            "intake_sha256": state.intake_digest,
            "baseline_plan_sha256": baseline_plan.digest() if baseline_plan is not None else None,
            "baseline_plan_core": project_strategy_plan(baseline_plan),
            "strategy_revision_context": {
                "precommit": precommit_strategy,
                "trigger": compact_event_for_model(trigger, excerpt_chars=1200) if trigger is not None else None,
                "agent_owned_generated_geometry": list(
                    self._agent_owned_generated_geometry_paths(state, baseline_plan)
                ),
                "case_files_committed": bool(case_files) and not precommit_strategy,
                "rule": (
                    "Python does not choose replacement geometry/meshing. For precommit revision, "
                    "change the EngineeringPlan only; the next author_case turn must author the full updated manifest."
                ),
            },
            "geometry_authoring_policy": self._geometry_authoring_policy(state),
            "current_case_files": case_files[-24:],
            "environment_hint": self.tools.environment_snapshot(),
            "tool_execution_contracts": self._mesh_tool_contracts(),
            "available_evidence": evidence_records[-4:],
            "bindings": bindings,
            "budget": budget,
        }
        instruction = (
            "Revise the engineering/meshing strategy that failed. If strategy_revision_context.precommit is true, "
            "no candidate case was committed: return revise_mesh_strategy with plan_patch (preferred) or updated_plan, "
            "change mesh_strategy/required_case_files and Agent-owned defaults as needed, and do not author files or "
            "native commands yet. If an Agent-generated surface was too large/incomplete, choose a self-contained "
            "implementation you can actually author on the next bounded turn. Preserve confirmed facts:\n"
        )
    elif contract_phase == "runtime_repair":
        payload: dict[str, object] = {
            "state_mode": "runtime_failure_slice",
            "phase": phase,
            "step": step,
            "confirmed_facts": [
                {"id": fact.id, "value": fact.value, "source": fact.source}
                for fact in (state.intake.facts if state.intake is not None else [])
                if fact.category != "context"
            ],
            "approved_plan": (
                {
                    "solver": state.engineering_plan.solver,
                    "execution": (state.engineering_plan.execution.model_dump(mode="json") if state.engineering_plan.execution else None),
                    "region_layouts": [x.model_dump(mode="json") for x in state.engineering_plan.region_layouts],
                    "interfaces": [x.model_dump(mode="json") for x in state.engineering_plan.interfaces],
                    "solver_provider_id": state.engineering_plan.solver_provider_id,
                    "temporal_behavior": state.engineering_plan.temporal_behavior,
                    "motion_kind": state.engineering_plan.motion_kind,
                    "mesh_motion_requirement": state.engineering_plan.mesh_motion_requirement,
                    "required_case_files": state.engineering_plan.required_case_files,
                    "confirmed_fact_bindings": [
                        item.model_dump(mode="json")
                        for item in state.engineering_plan.confirmed_fact_bindings
                    ],
                }
                if state.engineering_plan is not None else None
            ),
            "native_failure": self._redact_local_paths(runtime_log[-6000:]) if runtime_log else None,
            "case_file_contract_scan": self._runtime_case_file_contract_scan(state),
            "relevant_case_files": self._runtime_relevant_case_files(state, runtime_log),
            "mesh_evidence": {
                "passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
                "cell_count": state.mesh_evidence.cell_count if state.mesh_evidence else None,
                "manifest_current": bool(
                    self._checkmesh_mesh_manifest
                    and self._checkmesh_mesh_manifest == self.workspace.mesh_manifest_digest()
                ),
            },
            "available_evidence": evidence_records[-20:],
            "evidence_gap_status": self._evidence_gap_status("runtime_repair"),
            "bindings": bindings,
            "engineering_assumption_policy": assumption_policy,
            "evidence_retrieval_policy": retrieval_policy,
            "budget": budget,
        }
        instruction = (
            "Repair the actual runtime failure from this bounded diagnostic/file slice. "
            "Treat case_file_contract_scan as deterministic evidence: when it reports multiple "
            "invalid solve inputs, repair the whole systematic class of file-contract defects "
            "in one cycle rather than waiting for foamRun to fail on each file. "
            "Prefer repair_runtime_case. Use gather_evidence only when evidence_retrieval_policy.available "
            "is true and an explicit missing tool/version fact cannot be resolved from the supplied files/native diagnostic. "
            "If retrieval is unavailable, use existing evidence or block once with the correct block_kind:\n"
        )
    elif contract_phase == "repair":
        baseline_plan = self._pending_execution_plan or state.engineering_plan or self._draft_design_plan
        support_types = {
            "read_case_file", "read_reference", "search_references",
            "search_capabilities", "gather_evidence", "inspect_environment", "list_case_files",
        }
        recent_failures = [
            event for event in self._current_round_events(state)
            if (not event.success) and event.action_type not in support_types
            and (event.failure_category or "case") == "case"
        ]
        trigger = recent_failures[-1] if recent_failures else None
        diagnostic = ""
        if trigger is not None:
            diagnostic = "\n".join(
                part for part in (trigger.summary, trigger.output_excerpt) if str(part).strip()
            )
        relevant_files = self._validation_relevant_case_files(
            state,
            baseline_plan,
            diagnostic,
        )
        episode = (
            ensure_episode(state, trigger, [str(item.get("path") or "") for item in relevant_files if item.get("path")])
            if trigger is not None else state.repair_episode
        )
        observed_support = [
            compact_event_for_model(event, excerpt_chars=1400, summary_chars=500)
            for event in self._current_round_events(state)[-20:]
            if event.success and event.action_type in support_types
        ][-6:]
        episode_support = list(episode.supporting_observations[-6:]) if episode is not None else []
        supporting_observations = (episode_support + observed_support)[-6:]
        payload = {
            "state_mode": "case_validation_repair_v1",
            "phase": phase,
            "step": step,
            "confirmed_facts": [
                {"id": fact.id, "value": fact.value, "source": fact.source}
                for fact in (state.intake.facts if state.intake is not None else [])
                if fact.category != "context"
            ],
            "intake_sha256": state.intake_digest,
            "baseline_plan_sha256": baseline_plan.digest() if baseline_plan is not None else None,
            "baseline_plan_core": project_validation_repair_plan(baseline_plan),
            "repair_episode": (episode.model_dump(mode="json") if episode is not None else None),
            "validation_failure": (episode.current_failure if episode is not None else None),
            "relevant_case_files": relevant_files,
            "case_file_contract_scan": self._runtime_case_file_contract_scan(state),
            "case_inventory": case_files[:40],
            "mesh_evidence": {
                "passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
                "cell_count": state.mesh_evidence.cell_count if state.mesh_evidence else None,
                "max_non_orthogonality": state.mesh_evidence.max_non_orthogonality if state.mesh_evidence else None,
                "max_skewness": state.mesh_evidence.max_skewness if state.mesh_evidence else None,
                "manifest_current": bool(
                    self._checkmesh_mesh_manifest
                    and self._checkmesh_mesh_manifest == self.workspace.mesh_manifest_digest()
                ),
            },
            "supporting_evidence": evidence_records[-4:],
            "supporting_observations": supporting_observations,
            "bindings": bindings,
            "budget": budget,
            "repair_contract": {
                "mode": "failure_local_delta_only",
                "case_mutation_authority": "CaseDeltaGraph",
                "preserve_confirmed_intake": True,
                "preserve_unimplicated_engineering_choices": True,
                "controller_revalidates_after_delta": True,
                "read_results_return_on_next_repair_turn": True,
                "rule": (
                    "Fix the observed case validation failure, not the whole design. "
                    "Use plan_patch only if the diagnostic proves plan metadata must change."
                ),
            },
        }
        instruction = (
            "Repair one committed-case deterministic/native validation failure from this failure-local capsule. "
            "The controller has already supplied the current files most directly implicated by the diagnostic. "
            "Return repair_case_plan with the minimum justified delta, or block only if the observed case defect "
            "cannot be repaired without changing a confirmed user fact or unsupported capability. Do not redesign "
            "unrelated physics/geometry, do not repeat the complete EngineeringPlan, and do not regenerate unchanged "
            "files. For a missing fvSchemes/fvSolution dictionary section, choose the engineering/numerical content "
            "consistent with the supplied baseline plan and current file; Python will re-run CaseDeltaGraph and the "
            "actual OpenFOAM pre-solve consumer after the delta:\n"
        )
    elif use_delta:
        payload = {
            "state_mode": "delta_from_previous_response",
            "phase": phase,
            "step": step,
            "confirmed_facts": [
                {"id": fact.id, "value": fact.value, "source": fact.source}
                for fact in (state.intake.facts if state.intake is not None else [])
                if fact.category != "context"
            ],
            "baseline_plan": (
                {
                    "case_name": state.engineering_plan.case_name,
                    "solver": state.engineering_plan.solver,
                    "solver_provider_id": state.engineering_plan.solver_provider_id,
                    "openfoam_version": state.engineering_plan.openfoam_version,
                    "temporal_behavior": state.engineering_plan.temporal_behavior,
                    "motion_kind": state.engineering_plan.motion_kind,
                    "mesh_motion_requirement": state.engineering_plan.mesh_motion_requirement,
                    "required_case_files": state.engineering_plan.required_case_files,
                    "confirmed_fact_bindings": [
                        item.model_dump(mode="json")
                        for item in state.engineering_plan.confirmed_fact_bindings
                    ],
                    "confirmed_intake_sha256": state.engineering_plan.confirmed_intake_sha256,
                }
                if state.engineering_plan is not None
                else (
                    {
                        "solver": self._pending_execution_plan.solver,
                        "solver_provider_id": self._pending_execution_plan.solver_provider_id,
                        "required_case_files": self._pending_execution_plan.required_case_files,
                        "confirmed_fact_bindings": [
                            item.model_dump(mode="json")
                            for item in self._pending_execution_plan.confirmed_fact_bindings
                        ],
                        "confirmed_intake_sha256": self._pending_execution_plan.confirmed_intake_sha256,
                    }
                    if self._pending_execution_plan is not None else None
                )
            ),
            "bindings": bindings,
            "engineering_assumption_policy": assumption_policy,
            "evidence_retrieval_policy": retrieval_policy,
            "current_case_files": case_files,
            "recent_observations": self._recent_observations_for_model(state),
            "available_evidence": evidence_records,
            "evidence_gap_status": self._evidence_gap_status(phase),
            "tool_execution_contracts": self._mesh_tool_contracts(),
            "budget": budget,
            "ready_for_finalization": (
                self._ready_for_finalization(state, native_execution=native_execution)
                if phase in {"prepare", "prepare_finalize", "human_revision", "human_revision_finalize"}
                else False
            ),
            "runtime_log_excerpt": (
                self._redact_local_paths(runtime_log[-4000:]) if runtime_log else None
            ),
            "human_feedback": [
                {
                    "feedback_id": item.feedback_id,
                    "statement": item.statement,
                    "status": item.status,
                }
                for item in state.human_feedback[-self.policy.max_model_feedback_items :]
            ],
            "active_revision_proposal": (
                state.active_revision_proposal.model_dump(mode="json")
                if state.active_revision_proposal is not None else None
            ),
            "baseline_plan_decisions": (
                [item.model_dump(mode="json") for item in state.engineering_plan.decisions]
                if state.engineering_plan is not None and contract_phase == "revision" else []
            ),
        }
        if previous_snapshot.get("plan_sha256") != plan_digest:
            payload["current_engineering_plan"] = project_plan_core(state.engineering_plan, mode="generic")
            payload["current_engineering_plan_sha256"] = (state.engineering_plan.digest() if state.engineering_plan is not None else None)
        instruction = (
            "This is a compact state delta/capsule. Preserve confirmed_facts and baseline_plan; "
            "use recent evidence to return only the next changed action. Do not regenerate unchanged "
            "case content:\n"
        )
    else:
        payload = {
            "state_mode": "full",
            "phase": phase,
            "step": step,
            "confirmed_intake": project_confirmed_intake(state.intake),
            "intake_sha256": state.intake_digest,
            "exploratory_assumptions_authorized": state.user_request.exploratory_completion_authorized,
            "engineering_assumption_policy": assumption_policy,
            "evidence_retrieval_policy": retrieval_policy,
            "environment_hint": self.tools.environment_snapshot(),
            "capability_graph_hint": self.catalog.summary(),
            "preloaded_capability_providers": (
                self.catalog.search("", limit=self.policy.max_preloaded_capabilities)
                if self.policy.preload_capabilities else []
            ),
            "reference_roots": self.references.summary(),
            "current_case_files": case_files,
            "current_engineering_plan": project_plan_core(state.engineering_plan, mode="generic"),
            "current_engineering_plan_sha256": (state.engineering_plan.digest() if state.engineering_plan is not None else None),
            "pending_engineering_plan": project_plan_core(self._pending_execution_plan, mode="generic"),
            "pending_engineering_plan_sha256": (self._pending_execution_plan.digest() if self._pending_execution_plan is not None else None),
            "recent_observations": self._recent_observations_for_model(state),
            "cumulative_provenance": self._cumulative_provenance_summary(state),
            "available_evidence": evidence_records,
            "evidence_gap_status": self._evidence_gap_status(phase),
            "bindings": bindings,
            "deterministic_bindings": {
                "confirmed_intake": {
                    "bound_by": "python",
                    "sha256": state.intake_digest,
                    "fact_ids": sorted(
                        fact.id for fact in state.intake.facts if fact.category != "context"
                    ) if state.intake is not None else [],
                },
                "check_mesh": {
                    "bound_by": "python",
                    "passed": bool(state.mesh_evidence and state.mesh_evidence.passed),
                    "cell_count": state.mesh_evidence.cell_count if state.mesh_evidence else None,
                    "raw_log_sha256": state.mesh_evidence.raw_log_sha256 if state.mesh_evidence else None,
                },
                "case_manifest": {"bound_by": "python", "sha256": manifest_digest},
            },
            "tool_execution_contracts": self._mesh_tool_contracts(),
            "budget": budget,
            "ready_for_finalization": (
                self._ready_for_finalization(state, native_execution=native_execution)
                if phase in {"prepare", "prepare_finalize", "human_revision", "human_revision_finalize"}
                else False
            ),
            "human_feedback": project_feedback_history(state.human_feedback, limit=self.policy.max_model_feedback_items),
            "active_revision_proposal": project_revision_proposal(state.active_revision_proposal),
            "runtime_log_excerpt": (
                self._redact_local_paths(runtime_log[-4000:]) if runtime_log else None
            ),
        }
        if contract_phase == "prepare":
            instruction = (
                "Choose the next engineering action. Prefer execute_case_plan when ready. If an "
                "external tool/version fact is genuinely missing, declare one or more explicit gaps "
                "in a single gather_evidence batch; engineering-choice unknowns are assumptions, not "
                "retrieval gaps. When engineering_assumption_policy.authorized is true, choose reasonable "
                "representative missing values and record every concrete choice in plan.engineering_defaults "
                "instead of blocking merely because the user delegated those details:\n"
            )
        elif contract_phase == "prepare_decide":
            instruction = (
                "The bounded retrieval window is closed. Use the accumulated evidence and any authorized "
                "engineering defaults to return execute_case_plan. Block only when the physical objective or a "
                "tool/version requirement truly cannot be implemented without unsupported claims:\n"
            )
        elif contract_phase == "replan":
            instruction = (
                "The previous complete case bundle failed deterministic pre-commit authoring checks. "
                "The full candidate is retained in Python memory. Return only repair_candidate_case_plan "
                "with the minimum delta for the implicated candidate path(s), or block. No partial "
                "workspace case was committed:\n"
            )
        elif contract_phase == "block_mesh_replan":
            instruction = (
                "The retained candidate block_mesh failed deterministic topology preflight. Return one "
                "repair_candidate_block_mesh semantic replacement or block; do not text-patch it:\n"
            )
        elif contract_phase == "block_mesh_repair":
            instruction = (
                "Native blockMesh failed. Return one repair_block_mesh semantic replacement using the "
                "supplied structured_block_mesh baseline, or block:\n"
            )
        elif contract_phase == "strategy_revision":
            instruction = (
                "The current meshing strategy was invalidated by a deterministic tool contract or repeated identical native failure. "
                "Return revise_mesh_strategy with a different compatible meshing pipeline; do not retry the invalidated command unless its prerequisite state is explicitly changed:\n"
            )
        elif contract_phase == "repair":
            instruction = (
                "Return a delta-only repair. Prefer exact patches and do not repeat unchanged files "
                "or plan content. Use observed failure/evidence only:\n"
            )
        else:
            instruction = "Finalize or block from the validated state:\n"

    if contract_phase not in {"author_case", "prepare_design", "prepare_design_decide", "revision", "revision_decide", "revision_author", "strategy_revision", "repair"}:
        evidence_plan = self._pending_execution_plan or self._draft_design_plan or state.engineering_plan
        if self._pending_candidate_execution is not None:
            evidence_plan = self._pending_candidate_execution.plan
        payload["implementation_evidence_summary"] = advisory_authoring_evidence_summary(state, evidence_plan)
        instruction += " Observed syntax/reference evidence is advisory provenance. Use it when helpful, but missing syntax evidence alone must not block a file mutation; deterministic workspace, parser, native validation and safety gates decide whether the artifact can proceed. "

    if contract_phase in {"replan", "block_mesh_replan"}:
        payload["retained_candidate"] = self._candidate_repair_context()

    # Preserve the semantic blockMesh representation across native failures. A text
    # patch is intentionally not the primary repair interface for structured mesh
    # topology; when blockMesh itself failed, provide the exact TypedBlockMeshFile
    # that produced the authored dictionary so the Agent can return one complete
    # block_mesh replacement without whitespace-sensitive matching.
    recent_block_mesh_failure = any(
        (not event.success)
        and bool(event.failure_signature)
        and str(event.failure_signature).startswith("native:blockMesh:")
        for event in self._current_round_events(state)[-8:]
    )
    if (
        contract_phase in {"repair", "block_mesh_repair", "strategy_revision"}
        and recent_block_mesh_failure
        and self._structured_block_mesh is not None
    ):
        payload["structured_block_mesh"] = self._structured_block_mesh.model_dump(mode="json")

    if contract_phase == "author_case" and self._authoring_task_queue is not None:
        payload = self._authoring_task_queue["tasks"][self._authoring_task_queue["cursor"]]
    context_partition_metrics: dict[str, int] = {}
    prompt_char_limit = (
        self.policy.max_authoring_prompt_chars
        if contract_phase == "author_case"
        else self.policy.max_model_prompt_chars
    )
    try:
        if contract_phase == "repair":
            prompt_result, payload, context_partition_metrics = build_partitioned_validation_repair_prompt(
                instruction,
                payload,
                max_chars=prompt_char_limit,
            )
        else:
            prompt_result = build_bounded_json_prompt(instruction, payload, max_chars=prompt_char_limit)
    except ContextBudgetError:
        if contract_phase == "author_case":
            # The model-facing authoring capsule contains only a bounded projection of
            # the frozen plan. Task compilation, however, must remain bound to the
            # complete Python-held EngineeringPlan so accept_task can prove that the
            # controller authority did not change while file partitions were pending.
            # Rehydrate only inside the controller; compile_tasks immediately projects
            # each task again before any model call.
            partition_source = dict(payload)
            partition_source["frozen_engineering_plan"] = self._draft_design_plan.model_dump(mode="json")
            partition_source["confirmed_intake"] = state.intake.model_dump(mode="json") if state.intake is not None else None
            self._authoring_task_queue = compile_tasks(instruction, partition_source, prompt_char_limit)
            self.checkpoint(state, "authoring-partition-created")
            payload = self._authoring_task_queue["tasks"][0]
            prompt_result = build_bounded_json_prompt(instruction, payload, max_chars=prompt_char_limit)
        elif contract_phase in {"prepare_design", "prepare_design_decide"}:
            prompt_result, payload, context_partition_metrics = build_partitioned_design_prompt(
                instruction,
                payload,
                max_chars=self.policy.max_model_prompt_chars,
                initial_evidence_limit=len(evidence_records),
            )
            self.checkpoint(state, "design-context-partitioned")
        elif contract_phase == "revision":
            # build_partitioned_revision_prompt already performed every bounded
            # projection.  Preserve the explicit domain-specific diagnostic.
            raise
        elif contract_phase == "repair":
            # build_partitioned_validation_repair_prompt already exhausted only
            # optional supporting file/evidence projections. Preserve the
            # primary validation failure and refuse unsafe silent truncation.
            raise
        else:
            raise
    metrics = structured_request_metrics(
        schema,
        prompt_result.prompt,
        system_prompt=system_prompt,
    )
    metrics["compacted"] = prompt_result.compacted
    metrics["deltaContext"] = use_delta
    metrics["contractPhase"] = contract_phase
    metrics["evidenceShown"] = (
        len(payload.get("supporting_evidence", []))
        if contract_phase == "repair" and isinstance(payload, dict)
        else len(evidence_records)
    )
    metrics["evidenceObserved"] = evidence_total
    metrics["stagedAuthoring"] = bool(self.policy.staged_case_authoring)
    metrics["promptLimitChars"] = prompt_char_limit
    metrics.update(context_partition_metrics)
    model_name = getattr(self.llm, "model", None)
    if isinstance(model_name, str) and model_name:
        metrics["model"] = model_name
    max_output_tokens = getattr(self.llm, "max_output_tokens", None)
    if max_output_tokens is not None:
        metrics["maxOutputTokens"] = max_output_tokens
    self.progress.emit(
        ProgressEvent(
            phase="llm-context",
            message=f"{phase} LLM context 준비",
            status="info",
            step=local_step,
            limit=current_step_limit,
            metrics=metrics,
        )
    )

    kwargs = {
        "system_prompt": system_prompt,
    }
    if supports_stateful:
        kwargs.update(
            {
                "conversation_key": conversation_key,
                "use_previous_response": bool(use_delta and getattr(self.llm, "store", False)),
                "prompt_cache_key": f"ofa-eng-{contract_phase}",
            }
        )
    turn = self.llm.generate(schema, prompt_result.prompt, **kwargs)
    self._phase_prompt_counts[conversation_key] = prompt_count + 1
    feedback_digest = hashlib.sha256(
        json.dumps(
            [item.model_dump(mode="json") for item in state.human_feedback],
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    self._phase_context_snapshots[conversation_key] = {
        "plan_sha256": plan_digest,
        "manifest_sha256": manifest_digest,
        "feedback_sha256": feedback_digest,
    }
    usage = getattr(self.llm, "last_usage", None)
    if isinstance(usage, dict) and usage:
        usage = dict(usage)
        if isinstance(model_name, str) and model_name:
            usage["model"] = model_name
        self.progress.emit(
            ProgressEvent(
                phase="llm-usage",
                message=f"{phase} OpenAI token usage",
                status="info",
                step=local_step,
                limit=current_step_limit,
                metrics=usage,
            )
        )
    return turn

