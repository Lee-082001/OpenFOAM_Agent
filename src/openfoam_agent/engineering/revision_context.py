from __future__ import annotations

from copy import deepcopy

from openfoam_agent.llm.context import ContextBudgetError, PromptBuildResult, build_bounded_json_prompt, compact_text


def _compact_list_text(values: object, *, limit: int, item_chars: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in list(values or []):
        text = str(item or '').strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(compact_text(text, item_chars))
        if len(result) >= limit:
            break
    return result


def project_revision_plan(plan: object) -> dict[str, object] | None:
    """Project a sealed EngineeringPlan for delta revision without duplicating audit stores.

    The controller keeps the full EngineeringPlan in memory.  The model receives only
    decision-bearing fields needed to understand and patch the CFD design.  Immutable
    confirmed-fact/audit fields are represented by identity/bindings and are preserved
    by Python when ``plan_patch`` is applied.
    """
    if plan is None:
        return None
    raw = plan.model_dump(mode='json') if hasattr(plan, 'model_dump') else deepcopy(plan)
    if not isinstance(raw, dict):
        return None
    decisions = []
    for item in list(raw.get('decisions') or [])[-8:]:
        if not isinstance(item, dict):
            continue
        projected = {
            key: deepcopy(item.get(key))
            for key in ('area', 'risk_level', 'evidence_policy', 'verification_stage')
            if key in item
        }
        projected['choice'] = compact_text(str(item.get('choice') or ''), 180)
        if item.get('rationale'):
            projected['rationale'] = compact_text(str(item.get('rationale')), 120)
        decisions.append(projected)
    defaults = []
    for item in list(raw.get('engineering_defaults') or [])[:60]:
        if not isinstance(item, dict):
            continue
        projected = {
            key: deepcopy(item.get(key))
            for key in ('parameter', 'value', 'unit', 'basis', 'source')
            if key in item
        }
        defaults.append(projected)
    bindings = []
    for item in list(raw.get('confirmed_fact_bindings') or []):
        if not isinstance(item, dict):
            continue
        bindings.append({
            key: deepcopy(item.get(key))
            for key in ('fact_id', 'plan_fields', 'case_files', 'explanation')
            if key in item
        })
    return {
        'case_name': raw.get('case_name'),
        'solver': raw.get('solver'),
        'solver_provider_id': raw.get('solver_provider_id'),
        'execution': deepcopy(raw.get('execution')),
        'openfoam_distribution': raw.get('openfoam_distribution'),
        'openfoam_version': raw.get('openfoam_version'),
        'problem_interpretation': compact_text(str(raw.get('problem_interpretation') or ''), 1200),
        'temporal_behavior': raw.get('temporal_behavior'),
        'motion_kind': raw.get('motion_kind'),
        'mesh_motion_requirement': raw.get('mesh_motion_requirement'),
        'mesh_strategy': compact_text(str(raw.get('mesh_strategy') or ''), 1200),
        'region_layouts': deepcopy(raw.get('region_layouts') or []),
        'interfaces': deepcopy(raw.get('interfaces') or []),
        'completion': deepcopy(raw.get('completion')),
        'quantities_of_interest': deepcopy(raw.get('quantities_of_interest') or []),
        'conservation_checks': deepcopy(raw.get('conservation_checks') or []),
        'decisions': decisions,
        'assumptions': _compact_list_text(raw.get('assumptions'), limit=12, item_chars=260),
        'engineering_defaults': defaults,
        'required_case_files': list(raw.get('required_case_files') or []),
        'postprocess_strategy': _compact_list_text(raw.get('postprocess_strategy'), limit=30, item_chars=600),
        'confirmed_intake_sha256': raw.get('confirmed_intake_sha256'),
        'confirmed_fact_ids': list(raw.get('confirmed_fact_ids') or []),
        'confirmed_fact_bindings': bindings,
        'controller_preserved_fields': [
            'implementation_evidence_bindings',
            'evidence',
            'plan_conflicts',
        ],
        'projection_note': (
            'Decision/default rationales and audit evidence are compacted or omitted here only for model context; '
            'Python retains the complete sealed baseline and plan_patch changes only explicitly supplied fields.'
        ),
    }


def project_revision_proposal(proposal: object) -> dict[str, object] | None:
    if proposal is None:
        return None
    raw = proposal.model_dump(mode='json') if hasattr(proposal, 'model_dump') else deepcopy(proposal)
    if not isinstance(raw, dict):
        return None
    changes = []
    for item in list(raw.get('proposed_changes') or [])[:24]:
        if not isinstance(item, dict):
            continue
        changes.append({
            'area': compact_text(str(item.get('area') or ''), 160),
            'change': compact_text(str(item.get('change') or ''), 900),
            'rationale': compact_text(str(item.get('rationale') or ''), 700),
        })
    return {
        'proposal_id': raw.get('proposal_id'),
        'feedback_ids': list(raw.get('feedback_ids') or []),
        'diagnosis_summary': compact_text(str(raw.get('diagnosis_summary') or ''), 1800),
        'proposed_changes': changes,
        'expected_cost': raw.get('expected_cost'),
        'requires_case_revision': raw.get('requires_case_revision'),
        'requires_intake_revision': raw.get('requires_intake_revision'),
        'intake_revision_reason': compact_text(str(raw.get('intake_revision_reason') or ''), 800),
        'review_limitations': _compact_list_text(raw.get('review_limitations'), limit=12, item_chars=500),
        'baseline_plan_sha256': raw.get('baseline_plan_sha256'),
        'baseline_manifest_sha256': raw.get('baseline_manifest_sha256'),
    }


def project_revision_capsule(payload: dict[str, object], *, evidence_limit: int = 4, observation_limit: int = 3) -> dict[str, object]:
    evidence = list(payload.get('available_evidence') or [])[-max(0, evidence_limit):]
    observations = list(payload.get('recent_observations') or [])[-max(0, observation_limit):]
    return {
        'state_mode': 'human_revision_delta_v1',
        'phase': payload.get('phase'),
        'step': payload.get('step'),
        'confirmed_facts': deepcopy(payload.get('confirmed_facts') or []),
        'intake_sha256': payload.get('intake_sha256'),
        'baseline_plan_sha256': payload.get('baseline_plan_sha256'),
        'baseline_manifest_sha256': payload.get('baseline_manifest_sha256'),
        'baseline_plan_core': deepcopy(payload.get('baseline_plan_core')),
        'active_revision_proposal': deepcopy(payload.get('active_revision_proposal')),
        'feedback_observations': deepcopy(payload.get('feedback_observations') or []),
        'current_case_files': deepcopy(payload.get('current_case_files') or []),
        'mesh_evidence': deepcopy(payload.get('mesh_evidence')),
        'runtime_summary': deepcopy(payload.get('runtime_summary')),
        'recent_observations': observations,
        'available_evidence': evidence,
        'bindings': deepcopy(payload.get('bindings')),
        'budget': deepcopy(payload.get('budget')),
        'revision_contract': {
            'mode': 'delta_only',
            'plan_update_interface': 'prefer plan_patch; updated_plan is legacy compatibility only',
            'controller_preserves_immutable_plan_metadata': True,
            'case_mutation_authority': 'CaseDeltaGraph',
            'read_exact_files_on_demand': True,
        },
    }


def build_partitioned_revision_prompt(
    instruction: str,
    payload: dict[str, object],
    *,
    max_chars: int,
) -> tuple[PromptBuildResult, dict[str, object], dict[str, int]]:
    """Fit a human-revision prompt without re-sending the entire historical state."""
    attempts = [(4, 3), (2, 2), (1, 1), (0, 0)]
    last_error: ContextBudgetError | None = None
    for index, (evidence_limit, observation_limit) in enumerate(attempts, start=1):
        capsule = project_revision_capsule(
            payload,
            evidence_limit=evidence_limit,
            observation_limit=observation_limit,
        )
        try:
            result = build_bounded_json_prompt(instruction, capsule, max_chars=max_chars)
            return result, capsule, {
                'revisionPartitioned': 1,
                'revisionEvidenceLimit': evidence_limit,
                'revisionObservationLimit': observation_limit,
                'revisionPartitionAttempts': index,
            }
        except ContextBudgetError as exc:
            last_error = exc
    raise ContextBudgetError(
        'Human-feedback RevisionDeltaContext exceeds the engineering prompt budget even after optional '
        'evidence/observation removal. The sealed plan and confirmed intake remain unchanged; no case mutation '
        'was authorized or performed.'
    ) from last_error
