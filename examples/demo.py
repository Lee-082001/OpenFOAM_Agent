"""Offline v2 intake demo.

RuleBasedLLM deliberately stops before engineering; production case design requires
an autonomous model backend.
"""
from pathlib import Path
import json
import uuid

from openfoam_agent.llm.rule_based import RuleBasedLLM
from openfoam_agent.schemas.request import UserRequest
from openfoam_agent.workflow.engine import CFDWorkflow
from openfoam_agent.workflow.state import CFDState

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    request = UserRequest(
        prompt="사각형 장애물 주위 vortex shedding Re=1000 나머지는 탐색용으로 정해줘",
        exploratory_completion_authorized=True,
    )
    state = CFDState(run_id=str(uuid.uuid4()), user_request=request)
    workflow = CFDWorkflow(
        llm=RuleBasedLLM(),
        capability_db=ROOT / "config" / "openfoam14_capability_graph.json",
        workspace=ROOT / "workspace" / state.run_id,
        native_execution=False,
    )
    workflow.run(state)
    print(state.current_state.value)
    print(json.dumps(state.intake.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print("Confirming requires an autonomous engineering backend; there is no template fallback.")


if __name__ == "__main__":
    main()
