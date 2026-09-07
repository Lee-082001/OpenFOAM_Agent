from __future__ import annotations

from typing import Generic, Literal, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")

class AgentMessage(BaseModel, Generic[T]):
    status: Literal["success", "needs_input", "failed"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    data: T | None = None
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    next_action: str | None = None

class ToolResult(BaseModel):
    success: bool
    command: list[str]
    return_code: int
    stdout: str = ""
    stderr: str = ""

    cache_hit: bool = False
    termination_reason: str = "exited"
    log_path: str | None = None
    log_sha256: str | None = None
    output_bytes: int = 0
    output_truncated: bool = False
    wall_seconds: float | None = None
    process_id: int | None = None
    process_group_terminated: bool = False
