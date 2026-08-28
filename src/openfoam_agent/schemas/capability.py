from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ImplementationStrategy = Literal[
    "reuse",
    "dictionary",
    "fvModel",
    "custom_library",
    "derived_solver_module",
    "new_solver_module",
]
VerificationLevel = Literal[
    "unverified",
    "documented",
    "runtime_tested",
    "numerically_validated",
]

class CapabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["user_guide", "source_code", "runtime_test", "verification_report"]
    reference: str = Field(min_length=1)
    note: str = ""


class CapabilityProvider(BaseModel):
    """One evidence-bearing capability provider.

    A provider is data for the engineering agent, never a deterministic solver choice.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    provider_type: Literal["solver", "generated_solver", "model", "toolkit"]
    capabilities: list[str] = Field(default_factory=list)
    openfoam_version: str = Field(min_length=1)
    verified: bool = False
    verification_level: VerificationLevel = "unverified"
    evidence: list[CapabilityEvidence] = Field(default_factory=list)
    extension_points: list[ImplementationStrategy] = Field(default_factory=list)
    metadata: dict[str, str | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        if not self.capabilities:
            raise ValueError(f"Provider '{self.id}' has no capabilities.")
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError(f"Provider '{self.id}' contains duplicate capabilities.")
        if len(self.extension_points) != len(set(self.extension_points)):
            raise ValueError(f"Provider '{self.id}' contains duplicate extension points.")
        if self.verified and (self.verification_level == "unverified" or not self.evidence):
            raise ValueError(f"Verified provider '{self.id}' requires evidence and a level.")
        if not self.verified and self.verification_level != "unverified":
            raise ValueError(f"Unverified provider '{self.id}' cannot claim verification.")
        return self



class CapabilityGraphSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    graph_id: str = Field(min_length=1)
    openfoam_distribution: Literal["foundation"]
    openfoam_version: str = Field(min_length=1)
    providers: list[CapabilityProvider] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> Self:
        provider_ids = [provider.id for provider in self.providers]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("Capability graph contains duplicate provider IDs.")
        wrong_versions = [
            provider.id
            for provider in self.providers
            if provider.openfoam_version != self.openfoam_version
        ]
        if wrong_versions:
            raise ValueError(
                "Provider OpenFOAM version does not match graph version: "
                + ", ".join(sorted(wrong_versions))
            )
        return self
