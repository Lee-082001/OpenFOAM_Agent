from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from openfoam_agent.schemas.intake import CFDIntakeSpec
from openfoam_agent.schemas.request import UserRequest


class InteractionMode(StrEnum):
    EASY = "easy"
    GUIDED = "guided"
    STRICT = "strict"


_AUTHORIZATION_PATTERNS = (
    r"알아서",
    r"네가\s*(?:정|추천)",
    r"너가\s*(?:정|추천)",
    r"나머지\s*(?:전부|모두|싹다)?\s*(?:정|알아서)",
    r"탐색용으로\s*(?:정|해)",
    r"use (?:reasonable |sensible |recommended )?defaults?",
    r"choose (?:the )?(?:rest|remaining|defaults?)",
)
_REJECTION_PATTERNS = (
    r"알아서\s*(?:하지|정하지)\s*(?:마|말)",
    r"기본값(?:으로|을|은)?\s*(?:하지|쓰지|사용하지)\s*(?:마|말)",
    r"do not (?:use|choose) defaults?",
    r"don't (?:use|choose) defaults?",
)


def exploratory_authorization(text: str) -> bool | None:
    normalized = text.casefold()
    if any(re.search(pattern, normalized) for pattern in _REJECTION_PATTERNS):
        return False
    if any(re.search(pattern, normalized) for pattern in _AUTHORIZATION_PATTERNS):
        return True
    return None


@dataclass
class ConversationSession:
    mode: InteractionMode = InteractionMode.GUIDED
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[str] = field(default_factory=list)
    attempts: int = 0
    last_state: str | None = None
    pending_intake: CFDIntakeSpec | None = None
    confirmed_intake_digest: str | None = None
    pending_workflow_state: Any | None = field(default=None, repr=False)

    @property
    def exploratory_completion_authorized(self) -> bool:
        if self.mode == InteractionMode.STRICT:
            return False
        for turn in reversed(self.turns):
            decision = exploratory_authorization(turn)
            if decision is not None:
                return decision
        return self.mode == InteractionMode.EASY

    def add_turn(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Conversation turn must not be blank.")
        self.turns.append(normalized)
        self._invalidate_intake()

    def edit_last(self, text: str) -> None:
        if not self.turns:
            raise ValueError("There is no conversation turn to edit.")
        normalized = text.strip()
        if not normalized:
            raise ValueError("Replacement turn must not be blank.")
        self.turns[-1] = normalized
        self._invalidate_intake()

    def undo_last(self) -> str:
        if not self.turns:
            raise ValueError("There is no conversation turn to undo.")
        removed = self.turns.pop()
        self._invalidate_intake()
        return removed

    def reset(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.turns.clear()
        self.attempts = 0
        self.last_state = None
        self.pending_intake = None
        self.confirmed_intake_digest = None
        self.pending_workflow_state = None

    def _invalidate_intake(self) -> None:
        self.pending_intake = None
        self.confirmed_intake_digest = None
        self.pending_workflow_state = None

    def set_pending_intake(self, intake: CFDIntakeSpec | None) -> None:
        self.pending_intake = intake
        self.confirmed_intake_digest = None

    def next_attempt(self) -> int:
        self.attempts += 1
        return self.attempts

    def to_request(self) -> UserRequest:
        if not self.turns:
            raise ValueError("Add a prompt before running the conversation.")
        return UserRequest(
            prompt=self.turns[0],
            conversation_turns=self.turns[1:],
            interaction_mode=self.mode.value,
            exploratory_completion_authorized=self.exploratory_completion_authorized,
        )

    def summary(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "turns": list(self.turns),
            "exploratory_completion_authorized": self.exploratory_completion_authorized,
            "attempts": self.attempts,
            "last_state": self.last_state,
            "pending_intake": (
                {
                    "title": self.pending_intake.title,
                    "status": self.pending_intake.status,
                    "sha256": self.pending_intake.digest(),
                }
                if self.pending_intake
                else None
            ),
        }
