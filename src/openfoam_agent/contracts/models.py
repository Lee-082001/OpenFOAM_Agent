from __future__ import annotations

import math
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResourceLimits(Contract):
    max_native_processes: int = Field(default=40, ge=1, le=10000)
    wall_seconds: int = Field(default=3600, ge=1, le=86400)
    cpu_seconds: int | None = Field(default=None, ge=1, le=86400)
    memory_bytes: int | None = Field(default=None, ge=64 * 1024 * 1024)
    max_output_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    max_case_bytes: int = Field(default=2_000_000_000, ge=1024)
    max_ranks: int = Field(default=8, ge=1, le=256)


class ParallelExecution(Contract):
    mode: Literal["serial", "local_mpi"] = "serial"
    ranks: int = Field(default=1, ge=1, le=256)
    launcher: Literal["mpirun", "mpiexec"] = "mpirun"
    decomposition_method: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    reconstruct: bool = True

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "serial" and self.ranks != 1:
            raise ValueError("Serial execution requires exactly one rank.")
        if self.mode == "local_mpi" and (self.ranks < 2 or not self.decomposition_method):
            raise ValueError("Local MPI requires at least two ranks and an explicit decomposition method.")
        return self


class RegionCaseLayout(Contract):
    region: str = Field(default="", pattern=r"^(?:[A-Za-z][A-Za-z0-9_.-]*)?$")
    solver_module: str | None = None
    required_fields: list[str] = Field(default_factory=list)

    @property
    def field_dir(self) -> str:
        return f"0/{self.region}" if self.region else "0"

    @property
    def system_dir(self) -> str:
        return f"system/{self.region}" if self.region else "system"

    @property
    def constant_dir(self) -> str:
        return f"constant/{self.region}" if self.region else "constant"

    @property
    def mesh_dir(self) -> str:
        return f"{self.constant_dir}/polyMesh"

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        import re
        if self.region in {".", ".."}:
            raise ValueError("Unsafe region name.")
        if len(self.required_fields) != len(set(self.required_fields)):
            raise ValueError("Duplicate required region fields.")
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", x) for x in self.required_fields):
            raise ValueError("Required field names must be bare names.")
        return self


class RegionInterface(Contract):
    region: str
    patch: str
    neighbour_region: str
    neighbour_patch: str
    fields: list[str] = Field(default_factory=list)


class PhysicalQuantity(Contract):
    quantity: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    dimensions: tuple[int, int, int, int, int, int, int]
    region: str = ""
    patch: str = ""
    material: str = ""
    time_dependence: Literal["constant", "tabulated", "expression"] = "constant"
    reference_definition: str = ""

    @model_validator(mode="after")
    def finite_value(self) -> Self:
        if not math.isfinite(self.value):
            raise ValueError("Physical quantity must be finite.")
        return self


class ResidualThreshold(Contract):
    field: str = Field(min_length=1)
    value: float = Field(gt=0)


class ImplementationEvidenceBinding(Contract):
    path: str
    evidence_ids: list[str]


class CompletionContract(Contract):
    mode: Literal["transient", "steady", "custom"]
    start_time: float = 0.0
    end_time: float | None = None
    minimum_steps: int = Field(default=1, ge=1)
    residual_thresholds: list[ResidualThreshold] = Field(default_factory=list)
    consecutive_samples: int = Field(default=3, ge=1)
    required_result_fields: list[str] = Field(default_factory=list)
    custom_evidence: str | None = None

    @model_validator(mode="after")
    def validate_termination(self) -> Self:
        if not math.isfinite(self.start_time):
            raise ValueError("Completion start time must be finite.")
        if self.end_time is not None and (not math.isfinite(self.end_time) or self.end_time <= self.start_time):
            raise ValueError("Completion end time must exceed the start time.")
        if self.mode == "transient" and self.end_time is None:
            raise ValueError("Transient completion requires an end time.")
        if self.mode == "steady" and not self.residual_thresholds:
            raise ValueError("Steady completion requires explicit residual thresholds.")
        if any(not math.isfinite(v) or v <= 0 for v in (item.value for item in self.residual_thresholds)):
            raise ValueError("Residual thresholds must be finite and positive.")
        return self


class QuantityOfInterest(Contract):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    quantity: str = Field(min_length=1)
    region: str = ""
    selection: str = ""
    unit: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    time_column: int = Field(default=0, ge=0)
    value_column: int = Field(default=1, ge=0)
    operation: Literal["time_mean", "rms", "minimum", "maximum", "last", "integral", "difference", "balance"]
    other_source_path: str | None = None
    other_value_column: int = Field(default=1, ge=0)
    start_time: float | None = None
    end_time: float | None = None
    minimum_samples: int = Field(default=2, ge=1)
    expected_min: float | None = None
    expected_max: float | None = None


    @model_validator(mode="after")
    def validate_quantity_window(self) -> Self:
        for v in (self.start_time, self.end_time, self.expected_min, self.expected_max):
            if v is not None and not math.isfinite(v):
                raise ValueError("Quantity bounds must be finite.")
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise ValueError("Quantity time interval is reversed.")
        if self.expected_min is not None and self.expected_max is not None and self.expected_min > self.expected_max:
            raise ValueError("Quantity expected range is reversed.")
        if self.operation in {"difference", "balance"} and self.other_source_path is None:
            raise ValueError("Difference/balance needs two scalar data sources.")
        return self
