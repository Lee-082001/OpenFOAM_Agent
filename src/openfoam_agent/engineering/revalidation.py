from __future__ import annotations

from openfoam_agent.schemas.engineering import RevalidationRecord
from openfoam_agent.workflow.state import CFDState


def record_scoped_revalidation(
    state: CFDState,
    graph,
    *,
    phase: str,
    solver_retry_required: bool = False,
) -> RevalidationRecord:
    """Persist the controller's dependency-scoped revalidation decision.

    ``CaseDeltaGraph`` remains the authority for which native consumers are required.
    This helper only records that decision for audit/benchmarking; it never widens or
    narrows executable actions.
    """

    normalized_phase = phase
    if normalized_phase.startswith("revision"):
        normalized_phase = "strategy_revision"
    elif normalized_phase == "runtime_repair":
        normalized_phase = "runtime_repair"
    else:
        normalized_phase = "repair"

    record = RevalidationRecord(
        phase=normalized_phase,
        changed_paths=sorted(graph.changed_files),
        dropped_paths=sorted(graph.drop_paths),
        revalidation_domains=list(graph.revalidation_domains),
        affected_mesh_scopes=list(graph.affected_mesh_scope_keys),
        reused_mesh_scopes=list(graph.reused_mesh_scope_keys),
        dictionary_checks=len(graph.dictionary_paths),
        surface_checks=len(graph.surface_paths),
        native_commands=[
            " ".join([item.command, *item.arguments]).strip()
            for item in graph.native_pipeline
        ],
        pre_solve_required=bool(graph.validate_pre_solve),
        solver_retry_required=bool(solver_retry_required),
    )
    state.revalidation_records.append(record)
    return record
