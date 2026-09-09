from __future__ import annotations

import math
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ResourceLimits(Contract):
    max_native_processes: int = Field(default=40, ge=1, le=10000)
    wall_seconds: int = Field(default=3600, ge=1, le=86400)
    cpu_seconds: int | None = Field(default=None, ge=1, le=86400)
    memory_bytes: int | None = Field(default=None, ge=64 * 1024 * 1024)
    max_output_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    max_case_bytes: int = Field(default=2_000_000_000, ge=1024)
    max_ranks: int = Field(default=8, ge=1, le=256)
    max_processes: int = Field(default=256, ge=2, le=65536)
    cpu_cores: float = Field(default=8.0, gt=0, le=256, allow_inf_nan=False)
    total_cpu_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    total_wall_seconds: float = Field(default=86400.0, gt=0, allow_inf_nan=False)
    max_total_output_bytes: int = Field(default=1_000_000_000, ge=1024)


class ParallelRestart(Contract):
    time_name: str = Field(pattern=r"^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def finite_time(self) -> Self:
        if not math.isfinite(float(self.time_name)):
            raise ValueError("Restart time must be finite.")
        return self


class ParallelExecution(Contract):
    mode: Literal["serial", "local_mpi"] = "serial"
    ranks: int = Field(default=1, ge=1, le=256)
    launcher: Literal["mpirun", "mpiexec"] = "mpirun"
    decomposition_method: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    reconstruct: bool = True
    restart: ParallelRestart | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "serial" and self.restart is not None:
            raise ValueError("Parallel restart cannot be attached to serial execution.")
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
    resolution_state: Literal["intent", "resolved"] = "intent"
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
        if self.resolution_state == "resolved" and self.mode == "transient" and self.end_time is None:
            raise ValueError("Resolved transient completion requires an end time.")
        if self.resolution_state == "resolved" and self.mode == "steady" and not self.residual_thresholds:
            raise ValueError("Resolved steady completion requires explicit residual thresholds.")
        if any(not math.isfinite(v) or v <= 0 for v in (item.value for item in self.residual_thresholds)):
            raise ValueError("Residual thresholds must be finite and positive.")
        return self


class ExecutionBoundContract(Contract):
    """Controller-compiled upper bound for one approved native execution.

    This contract answers only "how is the process safely bounded and what output
    fields belong to this run?" It deliberately does not claim that the numerical
    solution is converged or physically acceptable.
    """

    mode: Literal["transient", "steady", "custom"]
    start_time: float = 0.0
    end_time: float | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    wall_seconds: int | None = Field(default=None, ge=1, le=86400)
    minimum_steps: int = Field(default=1, ge=1)
    required_result_fields: list[str] = Field(default_factory=list)
    source: Literal["plan_completion", "controlDict", "engineering_defaults", "mixed", "runtime_policy"] = "mixed"
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bound(self) -> Self:
        if not math.isfinite(self.start_time):
            raise ValueError("Execution-bound start time must be finite.")
        if self.end_time is not None and (not math.isfinite(self.end_time) or self.end_time <= self.start_time):
            raise ValueError("Execution-bound end time must exceed the start time.")
        if self.mode == "transient" and self.end_time is None and self.wall_seconds is None:
            raise ValueError("Transient execution requires an end time or bounded wall time.")
        if self.mode in {"steady", "custom"} and self.max_iterations is None and self.wall_seconds is None:
            raise ValueError("Steady/custom execution requires an iteration or wall-time bound.")
        return self


class ResultAcceptanceContract(Contract):
    """Post-execution numerical/physical acceptance criteria.

    Incomplete criteria do not block a bounded solve. They are carried forward as
    review warnings and can be evaluated after the native process has completed.
    """

    residual_thresholds: list[ResidualThreshold] = Field(default_factory=list)
    consecutive_samples: int = Field(default=3, ge=1)
    criteria_complete: bool = False
    advisory_criteria: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source: Literal["plan_completion", "engineering_defaults", "mixed", "none"] = "none"


class RuntimeContract(Contract):
    """Compiled runtime view derived from the immutable EngineeringPlan."""

    execution_bound: ExecutionBoundContract
    result_acceptance: ResultAcceptanceContract

class NativeFieldReduction(Contract):
    quantity_kind: Literal["temperature", "pressure", "density", "mass_flow", "volume_flow", "heat_rate", "mass", "energy"]
    field: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    resolution_state: Literal["intent", "resolved"] = "intent"
    time_names: list[str] = Field(default_factory=list, max_length=1000)
    reduction: Literal["patch_sum", "area_integral", "area_mean", "volume_integral", "volume_mean", "cell_minimum", "cell_maximum"]
    patches: list[str] = Field(default_factory=list, max_length=200)
    dimensions: tuple[int, int, int, int, int, int, int]

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        import re
        if self.resolution_state != "resolved":
            return self
        if any(not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", t) or not math.isfinite(float(t)) for t in self.time_names):
            raise ValueError("Field observation requires literal finite time directory names.")
        times = [float(t) for t in self.time_names]
        if times != sorted(set(times)):
            raise ValueError("Field observation times must be strictly increasing and unique.")
        if len(self.patches) != len(set(self.patches)) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", p) for p in self.patches):
            raise ValueError("Patch selections must be unique literal names, not regexes.")
        surface = self.reduction in {"patch_sum", "area_integral", "area_mean"}
        if surface != bool(self.patches):
            raise ValueError("Surface reduction requires explicit patches; volume reduction uses the whole region.")
        return self


class RegionBalance(Contract):
    region: str = Field(default="", pattern=r"^(?:[A-Za-z][A-Za-z0-9_.-]*)?$")
    flux_field: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    storage_mode: Literal["steady", "density_field"]
    storage_density_field: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    source_mode: Literal["zero", "density_field"]
    source_density_field: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        # RegionBalance can be emitted as design intent before result fields exist.
        # Normalize harmless mode/field disagreements instead of rejecting the whole plan.
        if self.storage_density_field and self.storage_mode != "density_field":
            self.storage_mode = "density_field"
        elif not self.storage_density_field and self.storage_mode == "density_field":
            self.storage_mode = "steady"
        if self.source_density_field and self.source_mode != "density_field":
            self.source_mode = "density_field"
        elif not self.source_density_field and self.source_mode == "density_field":
            self.source_mode = "zero"
        return self


class ConservationCheck(Contract):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    kind: Literal["mass", "energy", "volume"]
    resolution_state: Literal["intent", "resolved"] = "intent"
    regions: list[RegionBalance] = Field(default_factory=list, max_length=30)
    time_names: list[str] = Field(default_factory=list, max_length=1000)
    absolute_tolerance: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    relative_tolerance: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    check_interfaces: bool = True

    @model_validator(mode="after")
    def validate_balance(self) -> Self:
        if self.resolution_state != "resolved":
            return self
        NativeFieldReduction(quantity_kind="temperature", field="probe", resolution_state="resolved", time_names=self.time_names, reduction="volume_mean", dimensions=(0,0,0,0,0,0,0))
        if not self.regions:
            raise ValueError("Resolved conservation check requires at least one region.")
        if len({r.region for r in self.regions}) != len(self.regions):
            raise ValueError("Conservation region names must be unique.")
        return self


class QuantityOfInterest(Contract):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    quantity: str = Field(min_length=1)
    resolution_state: Literal["intent", "resolved"] = "intent"
    region: str = ""
    selection: str = ""
    unit: str = Field(default="unknown", min_length=1)
    source_path: str = ""
    native_field: NativeFieldReduction | None = None
    time_column: int = Field(default=0, ge=0)
    value_column: int = Field(default=1, ge=0)
    operation: Literal["time_mean", "rms", "minimum", "maximum", "last", "integral", "difference", "balance"] = "last"
    other_source_path: str | None = None
    other_value_column: int = Field(default=1, ge=0)
    start_time: float | None = None
    end_time: float | None = None
    minimum_samples: int = Field(default=2, ge=1)
    expected_min: float | None = None
    expected_max: float | None = None


    @model_validator(mode="after")
    def validate_quantity_window(self) -> Self:
        if self.resolution_state == "resolved":
            if self.native_field is None and not self.source_path:
                raise ValueError("Resolved quantity requires a scalar table source or native field observation.")
            if self.native_field is not None and (self.source_path or self.operation in {"difference", "balance"} or self.other_source_path):
                raise ValueError("Resolved native-field analysis cannot mix table sources or table-difference operations.")
        for v in (self.start_time, self.end_time, self.expected_min, self.expected_max):
            if v is not None and not math.isfinite(v):
                raise ValueError("Quantity bounds must be finite.")
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise ValueError("Quantity time interval is reversed.")
        if self.expected_min is not None and self.expected_max is not None and self.expected_min > self.expected_max:
            raise ValueError("Quantity expected range is reversed.")
        if self.resolution_state == "resolved" and self.operation in {"difference", "balance"} and self.other_source_path is None:
            raise ValueError("Resolved difference/balance needs two scalar data sources.")
        return self
