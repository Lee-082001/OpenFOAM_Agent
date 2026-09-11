from __future__ import annotations

import json

from openfoam_agent.llm.context import compact_event_for_model
from openfoam_agent.schemas.engineering import EngineeringEvent, RepairEpisode
from openfoam_agent.workflow.state import CFDState

_SUPPORT_ACTIONS = {
    "read_case_file", "read_reference", "search_references", "search_capabilities",
    "gather_evidence", "inspect_environment", "list_case_files",
}
_SUPPORT_ARCHIVE_LIMIT = 24
_REPAIR_HISTORY_ARCHIVE_LIMIT = 24
_REFERENCE_SUPPORT_ACTIONS = {"read_reference", "search_references", "search_capabilities", "gather_evidence"}


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
        episode.previous_repairs = episode.previous_repairs[-_REPAIR_HISTORY_ARCHIVE_LIMIT:]
        episode.current_failure = record
        episode.validation_generation += 1
    episode.current_implicated_files = list(dict.fromkeys(implicated_files))[:24]
    return episode


def add_support_observation(state: CFDState, event: EngineeringEvent) -> None:
    """Archive bounded support evidence without making the archive prompt memory."""
    if state.repair_episode is None or not event.success or event.action_type not in _SUPPORT_ACTIONS:
        return
    # Reference searches are particularly prone to returning many hits. Keep a
    # smaller event digest in the durable episode; repair_context.py applies a
    # second, stricter top-k/character projection before any model call.
    if event.action_type in _REFERENCE_SUPPORT_ACTIONS:
        record = compact_event_for_model(event, excerpt_chars=700, summary_chars=280)
    else:
        record = compact_event_for_model(event, excerpt_chars=1000, summary_chars=400)
    token = json.dumps(record, ensure_ascii=True, sort_keys=True, default=str)
    existing = {
        json.dumps(item, ensure_ascii=True, sort_keys=True, default=str)
        for item in state.repair_episode.supporting_observations[-_SUPPORT_ARCHIVE_LIMIT:]
    }
    if token not in existing:
        state.repair_episode.supporting_observations.append(record)
    state.repair_episode.supporting_observations = (
        state.repair_episode.supporting_observations[-_SUPPORT_ARCHIVE_LIMIT:]
    )


def clear_episode(state: CFDState) -> None:
    state.repair_episode = None


def record_repair(state: CFDState, *, diagnosis: str, changed_files: list[str]) -> None:
    episode = state.repair_episode
    if episode is None:
        return
    episode.previous_repairs.append({
        "generation": episode.validation_generation,
        "diagnosis": diagnosis[:900],
        "changed_files": list(dict.fromkeys(changed_files))[:16],
    })
    episode.previous_repairs = episode.previous_repairs[-_REPAIR_HISTORY_ARCHIVE_LIMIT:]
