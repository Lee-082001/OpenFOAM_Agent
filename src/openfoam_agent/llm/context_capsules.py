from __future__ import annotations

from copy import deepcopy
from typing import Any

from openfoam_agent.llm.context import compact_text


def _dump(obj: object) -> dict[str, Any] | None:
    if obj is None:
        return None
    if hasattr(obj, 'model_dump'):
        data = obj.model_dump(mode='json')
    elif isinstance(obj, dict):
        data = deepcopy(obj)
    else:
        return None
    return data if isinstance(data, dict) else None


def project_confirmed_intake(intake: object, *, max_facts: int = 80) -> dict[str, Any] | None:
    raw = _dump(intake)
    if raw is None:
        return None
    facts = []
    for item in list(raw.get('facts') or [])[:max_facts]:
        if not isinstance(item, dict):
            continue
        facts.append({
            'id': item.get('id'),
            'value': compact_text(str(item.get('value') or ''), 700),
            'source': item.get('source'),
            'category': item.get('category'),
        })
    return {
        'summary': compact_text(str(raw.get('summary') or ''), 1200),
        'status': raw.get('status'),
        'facts': facts,
        'blocking_unknowns': [compact_text(str(x), 400) for x in list(raw.get('blocking_unknowns') or [])[:20]],
    }


def project_plan_core(plan: object, *, mode: str = 'generic') -> dict[str, Any] | None:
    raw = _dump(plan)
    if raw is None:
        return None
    defaults_limit = {'authoring': 32, 'revision': 16, 'repair': 16}.get(mode, 16)
    decisions_limit = {'authoring': 12, 'revision': 6}.get(mode, 8)
    defaults = []
    for item in list(raw.get('engineering_defaults') or [])[:defaults_limit]:
        if not isinstance(item, dict):
            continue
        defaults.append({
            'parameter': compact_text(str(item.get('parameter') or ''), 140),
            'value': compact_text(str(item.get('value') or ''), 240),
            'unit': compact_text(str(item.get('unit') or ''), 80),
            'basis': item.get('basis'),
        })
    decisions = []
    for item in list(raw.get('decisions') or [])[-decisions_limit:]:
        if not isinstance(item, dict):
            continue
        decisions.append({
            'area': compact_text(str(item.get('area') or ''), 120),
            'choice': compact_text(str(item.get('choice') or ''), 280),
            'rationale': compact_text(str(item.get('rationale') or ''), 180),
            'verification_stage': item.get('verification_stage'),
        })
    bindings = []
    for item in list(raw.get('confirmed_fact_bindings') or [])[:120]:
        if not isinstance(item, dict):
            continue
        bindings.append({
            'fact_id': item.get('fact_id'),
            'plan_fields': list(item.get('plan_fields') or [])[:12],
            'case_files': list(item.get('case_files') or [])[:12],
        })
    out = {
        'case_name': raw.get('case_name'),
        'solver': raw.get('solver'),
        'solver_provider_id': raw.get('solver_provider_id'),
        'execution': deepcopy(raw.get('execution')),
        'openfoam_distribution': raw.get('openfoam_distribution'),
        'openfoam_version': raw.get('openfoam_version'),
        'problem_interpretation': compact_text(str(raw.get('problem_interpretation') or ''), 1000),
        'temporal_behavior': raw.get('temporal_behavior'),
        'motion_kind': raw.get('motion_kind'),
        'mesh_motion_requirement': raw.get('mesh_motion_requirement'),
        'mesh_strategy': compact_text(str(raw.get('mesh_strategy') or ''), 1000),
        'region_layouts': deepcopy(raw.get('region_layouts') or [])[:24],
        'interfaces': deepcopy(raw.get('interfaces') or [])[:24],
        'completion': deepcopy(raw.get('completion')),
        'engineering_defaults': defaults,
        'decisions': decisions,
        'assumptions': [compact_text(str(x), 220) for x in list(raw.get('assumptions') or [])[:(8 if mode == 'revision' else 16)]],
        'required_case_files': list(raw.get('required_case_files') or [])[:80],
        'confirmed_intake_sha256': raw.get('confirmed_intake_sha256'),
        'confirmed_fact_ids': list(raw.get('confirmed_fact_ids') or [])[:200],
        'confirmed_fact_bindings': bindings,
    }
    if mode in {'postprocess','review'}:
        out['quantities_of_interest'] = deepcopy(raw.get('quantities_of_interest') or [])[:32]
        out['conservation_checks'] = deepcopy(raw.get('conservation_checks') or [])[:32]
        out['postprocess_strategy'] = [compact_text(str(x), 500) for x in list(raw.get('postprocess_strategy') or [])[:32]]
    return out


def project_feedback_history(items: list[object], *, limit: int = 8) -> list[dict[str, Any]]:
    out=[]
    for item in items[-limit:]:
        raw=_dump(item)
        if raw is None: continue
        out.append({
            'feedback_id': raw.get('feedback_id'), 'scope': raw.get('scope'),
            'statement': compact_text(str(raw.get('statement') or ''), 1000),
            'submitted_state': raw.get('submitted_state'), 'status': raw.get('status'),
        })
    return out


def project_revision_history(items: list[object], *, limit: int = 4) -> list[dict[str, Any]]:
    out=[]
    for item in items[-limit:]:
        raw=_dump(item)
        if raw is None: continue
        out.append({
            'revision_id': raw.get('revision_id'), 'proposal_id': raw.get('proposal_id'),
            'summary': compact_text(str(raw.get('summary') or raw.get('diagnosis_summary') or ''), 700),
            'archive_path': raw.get('archive_path'),
        })
    return out
