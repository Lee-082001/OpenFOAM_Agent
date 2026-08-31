from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResidualSample(_RuntimeModel):
    time: float
    field: str = Field(min_length=1)
    initial_residual: float
    final_residual: float


class RuntimePolicy(_RuntimeModel):
    # One initial solver execution plus up to eight autonomous repair/retry cycles.
    max_attempts: int = Field(default=9, ge=1, le=13)
    solver_timeout_seconds: int = Field(default=3600, ge=1, le=86_400)

    @property
    def max_repair_cycles(self) -> int:
        return max(0, self.max_attempts - 1)


class SimulationResult(_RuntimeModel):
    success: bool
    completed: bool
    return_code: int
    last_time: float | None = None
    residuals: list[ResidualSample] = Field(default_factory=list)
    courant_max: float | None = None
    continuity_error: float | None = None
    fatal_error: str | None = None
    non_finite_detected: bool = False
    end_marker_found: bool = False
    log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_failures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_success_claim(self) -> Self:
        if self.success and (
            not self.completed
            or self.return_code != 0
            or self.fatal_error is not None
            or self.non_finite_detected
            or not self.end_marker_found
            or self.evidence_failures
        ):
            raise ValueError("Successful runtime result requires passing execution evidence.")
        for value in (self.last_time, self.courant_max, self.continuity_error):
            if value is not None and not math.isfinite(value):
                raise ValueError("Runtime metrics must be finite when present.")
        return self


class SimulationAttempt(_RuntimeModel):
    attempt: int = Field(ge=1, le=13)
    result: SimulationResult
    repair_requested: bool = False


class RuntimeReport(_RuntimeModel):
    success: bool
    attempts: list[SimulationAttempt] = Field(default_factory=list)
    final_result: SimulationResult

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if not self.attempts:
            raise ValueError("Runtime report requires at least one attempt.")
        expected = list(range(1, len(self.attempts) + 1))
        if [item.attempt for item in self.attempts] != expected:
            raise ValueError("Runtime attempts must be consecutive.")
        if any(item.result.success for item in self.attempts[:-1]):
            raise ValueError("No attempt may follow a successful runtime result.")
        if self.final_result != self.attempts[-1].result:
            raise ValueError("Runtime final result must match the last attempt.")
        if self.success != self.final_result.success:
            raise ValueError("Runtime report success disagrees with final result.")
        return self
