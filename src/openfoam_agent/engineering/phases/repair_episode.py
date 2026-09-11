from __future__ import annotations

from openfoam_agent.llm.context import compact_event_for_model
from openfoam_agent.schemas.engineering import EngineeringEvent, RepairEpisode
from openfoam_agent.workflow.state import CFDState

_SUPPORT_ACTIONS = {
    "read_case_file", "read_reference", "search_references", "search_capabilities",
    "gather_evidence", "inspect_environment", "list_case_files",
}


def ensure_episode(state: CFDState, failure: EngineeringEvent, implicated_files: list[str]) -> RepairEpisode:
    record = compact_event_for_model(failure, excerpt_chars=4200, summary_chars=1200)
    episode = state.repair_episode
    if episode is None:
        episode = RepairEpisode(
            episode_id=f"repair-{len(state.secondary_failures)+len(state.engineering_events)+1:04d}",
            root_failure=record,
            current_failure=record,
            current_implicated_files=list(dict.fromkeys(implicated_files))[:24],
        )
        state.repair_episode = episode
        return episode

    # Only a new failing validation/native event advances current_failure. Support
    # reads/searches can never overwrite the active diagnostic.
    previous = episode.current_failure
    if previous != record:
        # The prior diagnostic remains root/history; actual repair deltas are
        # recorded separately by record_repair().
        episode.previous_repairs = episode.previous_repairs[-24:]
        episode.current_failure = record
        episode.validation_generation += 1
    episode.current_implicated_files = list(dict.fromkeys(implicated_files))[:24]
    return episode


def add_support_observation(state: CFDState, event: EngineeringEvent) -> None:
    if state.repair_episode is None or not event.success or event.action_type not in _SUPPORT_ACTIONS:
        return
    state.repair_episode.supporting_observations.append(
        compact_event_for_model(event, excerpt_chars=1400, summary_chars=500)
    )
    state.repair_episode.supporting_observations = state.repair_episode.supporting_observations[-24:]


def clear_episode(state: CFDState) -> None:
    state.repair_episode = None


def record_repair(state: CFDState, *, diagnosis: str, changed_files: list[str]) -> None:
    episode = state.repair_episode
    if episode is None:
        return
    episode.previous_repairs.append({
        "generation": episode.validation_generation,
        "diagnosis": diagnosis[:1600],
        "changed_files": list(dict.fromkeys(changed_files))[:24],
    })
    episode.previous_repairs = episode.previous_repairs[-24:]
