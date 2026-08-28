from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    geometry_files: list[str] = Field(default_factory=list)
    additional_files: list[str] = Field(default_factory=list)
    conversation_turns: list[str] = Field(default_factory=list)
    interaction_mode: Literal["easy", "guided", "strict"] = "guided"
    exploratory_completion_authorized: bool = False

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("CFD request prompt must not be blank.")
        return value

    @field_validator("conversation_turns", "geometry_files", "additional_files")
    @classmethod
    def reject_blank_items(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Request list entries must not be blank.")
        return normalized
