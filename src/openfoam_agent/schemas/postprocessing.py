from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openfoam_agent.schemas.engineering import TypedFoamDictionaryFile


class _PostModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PostProcessingArtifact(_PostModel):
    kind: Literal[
        "vorticity_field",
        "force_coefficients",
        "forces",
        "sampled_data",
        "other",
    ]
    path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ForceCoefficientAnalysis(_PostModel):
    source_path: str = Field(min_length=1, max_length=500)
    dictionary_path: str = Field(min_length=1, max_length=500)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dictionary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples_total: int = Field(ge=0)
    samples_used: int = Field(ge=0)
    discard_fraction: float = Field(ge=0.0, lt=0.9)
    start_time: float | None = None
    end_time: float | None = None
    mean_cd: float | None = None
    mean_cl: float | None = None
    rms_cl: float | None = None
    shedding_frequency: float | None = Field(default=None, ge=0.0)
    reference_velocity: float | None = Field(default=None, gt=0.0)
    reference_length: float | None = Field(default=None, gt=0.0)
    strouhal_number: float | None = Field(default=None, ge=0.0)
    periods_observed: int = Field(default=0, ge=0)
    period_cv: float | None = Field(default=None, ge=0.0)
    limitations: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_finite_metrics(self) -> Self:
        values = (
            self.start_time,
            self.end_time,
            self.mean_cd,
            self.mean_cl,
            self.rms_cl,
            self.shedding_frequency,
            self.reference_velocity,
            self.reference_length,
            self.strouhal_number,
            self.period_cv,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("Post-processing metrics must be finite when present.")
        if self.strouhal_number is not None and (
            self.shedding_frequency is None
            or self.reference_velocity is None
            or self.reference_length is None
        ):
            raise ValueError("Strouhal evidence requires frequency and reference scales.")
        return self


class PostProcessingReport(_PostModel):
    success: bool
    summary: str = Field(min_length=1, max_length=4000)
    scientific_confidence: Literal["unknown", "low", "moderate", "high"] = "unknown"
    review_reasons: list[str] = Field(default_factory=list, max_length=40)
    recommended_human_checks: list[str] = Field(default_factory=list, max_length=40)
    artifacts: list[PostProcessingArtifact] = Field(default_factory=list, max_length=200)
    force_analysis: ForceCoefficientAnalysis | None = None
    limitations: list[str] = Field(default_factory=list, max_length=50)
    actions_executed: int = Field(default=0, ge=0)
    native_commands_executed: int = Field(default=0, ge=0)


class SearchPostProcessReferencesAction(_PostModel):
    type: Literal["search_postprocess_references"]
    query: str = Field(min_length=1, max_length=500)
    scope: Literal["all", "tutorials", "source", "etc"] = "all"
    rationale: str = Field(default="", max_length=200)


class ReadPostProcessReferenceAction(_PostModel):
    type: Literal["read_postprocess_reference"]
    reference: str = Field(min_length=1, max_length=1000)
    start_line: int = Field(default=1, ge=1, le=1_000_000)
    line_count: int = Field(default=160, ge=1, le=400)
    rationale: str = Field(default="", max_length=200)


class WritePostProcessConfigAction(_PostModel):
    type: Literal["write_postprocess_config"]
    path: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=1_000_000)
    rationale: str = Field(default="", max_length=200)


class RunFoamPostProcessAction(_PostModel):
    type: Literal["run_foam_postprocess"]
    dictionary_path: str = Field(min_length=1, max_length=240)
    time_selection: Literal["all", "latest"] = "all"
    use_solver_context: bool = True
    rationale: str = Field(default="", max_length=200)


class ListResultFilesAction(_PostModel):
    type: Literal["list_result_files"]
    prefix: str = Field(default="", max_length=240)
    rationale: str = Field(default="", max_length=200)


class ReadResultFileAction(_PostModel):
    type: Literal["read_result_file"]
    path: str = Field(min_length=1, max_length=500)
    max_chars: int = Field(default=40_000, ge=1, le=120_000)
    rationale: str = Field(default="", max_length=200)


class AnalyzeForceCoefficientsAction(_PostModel):
    type: Literal["analyze_force_coefficients"]
    coefficient_path: str = Field(min_length=1, max_length=500)
    dictionary_path: str = Field(min_length=1, max_length=240)
    discard_fraction: float = Field(default=0.25, ge=0.0, lt=0.9)
    rationale: str = Field(default="", max_length=200)


class PostProcessConfigFile(_PostModel):
    path: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=300_000)


class PostProcessRunSpec(_PostModel):
    dictionary_path: str = Field(min_length=1, max_length=240)
    time_selection: Literal["all", "latest"] = "all"
    use_solver_context: bool = True


class ForceAnalysisSpec(_PostModel):
    coefficient_path: str = Field(min_length=1, max_length=500)
    dictionary_path: str = Field(min_length=1, max_length=240)
    discard_fraction: float = Field(default=0.25, ge=0.0, lt=0.9)


class PostProcessingExecutionPlanAction(_PostModel):
    """Execute predictable post-processing work without an LLM turn per tool."""

    type: Literal["execute_postprocessing_plan"]
    goal: str = Field(min_length=1, max_length=500)
    configs: list[PostProcessConfigFile] = Field(default_factory=list, max_length=12)
    typed_configs: list[TypedFoamDictionaryFile] = Field(default_factory=list, max_length=12)
    runs: list[PostProcessRunSpec] = Field(default_factory=list, max_length=12)
    force_analyses: list[ForceAnalysisSpec] = Field(default_factory=list, max_length=8)
    summary: str = Field(min_length=1, max_length=1500)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    scientific_confidence: Literal["unknown", "low", "moderate", "high"] = "unknown"
    review_reasons: list[str] = Field(default_factory=list, max_length=24)
    recommended_human_checks: list[str] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if not (self.configs or self.typed_configs or self.runs or self.force_analyses):
            raise ValueError("Post-processing execution plan must contain deterministic work.")
        paths = [x.path for x in self.configs] + [x.path for x in self.typed_configs]
        if len(paths) != len(set(paths)):
            raise ValueError("Post-processing plan contains duplicate config paths.")
        for path in paths:
            if not path.startswith("postprocessConfig/") or ".." in path:
                raise ValueError("Post-processing config must live under postprocessConfig/.")
        return self


class FinishPostProcessingAction(_PostModel):
    type: Literal["finish_postprocessing"]
    summary: str = Field(min_length=1, max_length=4000)
    limitations: list[str] = Field(default_factory=list, max_length=50)
    scientific_confidence: Literal["unknown", "low", "moderate", "high"] = "unknown"
    review_reasons: list[str] = Field(default_factory=list, max_length=40)
    recommended_human_checks: list[str] = Field(default_factory=list, max_length=40)
    rationale: str = Field(default="", max_length=200)


class BlockPostProcessingAction(_PostModel):
    type: Literal["block_postprocessing"]
    reason: str = Field(min_length=1, max_length=4000)
    rationale: str = Field(default="", max_length=200)


PostProcessingAction = (
    SearchPostProcessReferencesAction
    | ReadPostProcessReferenceAction
    | WritePostProcessConfigAction
    | RunFoamPostProcessAction
    | ListResultFilesAction
    | ReadResultFileAction
    | AnalyzeForceCoefficientsAction
    | PostProcessingExecutionPlanAction
    | FinishPostProcessingAction
    | BlockPostProcessingAction
)


CompactPostProcessingAction = (
    SearchPostProcessReferencesAction
    | ReadPostProcessReferenceAction
    | ReadResultFileAction
    | PostProcessingExecutionPlanAction
    | BlockPostProcessingAction
)


class PostProcessingPlanTurn(_PostModel):
    action: CompactPostProcessingAction


class PostProcessingTurn(_PostModel):
    action: PostProcessingAction


class PostProcessingEvent(_PostModel):
    step: int = Field(ge=1)
    action_type: str = Field(min_length=1, max_length=80)
    success: bool
    summary: str = Field(min_length=1, max_length=4000)
    output_excerpt: str = Field(default="", max_length=12000)
    native_command_executed: bool = False
    artifact_path: str | None = Field(default=None, max_length=500)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> Self:
        if (self.artifact_path is None) != (self.artifact_sha256 is None):
            raise ValueError("Post-processing artifact path/hash must be recorded together.")
        return self
