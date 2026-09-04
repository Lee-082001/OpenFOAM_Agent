from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InstalledExecutable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.+-]*$", max_length=160)
    category: Literal["solver_application", "execution_driver", "utility", "script", "unknown"] = "unknown"
    trusted: bool = True


class InstalledComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.+-]*$", max_length=160)
    category: Literal["solver_module", "fv_model", "function_object", "source_component"]
    source: Literal["installed_source", "documented_profile"]


class InstalledOpenFOAMIR(BaseModel):
    """Sanitized semantic inventory of one sourced OpenFOAM Foundation installation.

    Absolute installation paths are deliberately excluded.  The IR records only names and
    provenance needed for engineering decisions.  Executable trust is re-checked by
    SafeRunner immediately before every process launch.
    """

    model_config = ConfigDict(extra="forbid")

    distribution: Literal["foundation"] = "foundation"
    version: Literal["13", "14"] | None = None
    installation_configured: bool = False
    executables: list[InstalledExecutable] = Field(default_factory=list, max_length=1000)
    components: list[InstalledComponent] = Field(default_factory=list, max_length=1000)
    source_scopes: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_unique_names(self):
        exe_names = [item.name for item in self.executables]
        if len(exe_names) != len(set(exe_names)):
            raise ValueError("Installed OpenFOAM IR contains duplicate executable names.")
        component_keys = [(item.category, item.name) for item in self.components]
        if len(component_keys) != len(set(component_keys)):
            raise ValueError("Installed OpenFOAM IR contains duplicate components.")
        return self

    @property
    def executable_names(self) -> set[str]:
        return {item.name for item in self.executables}

    @property
    def solver_modules(self) -> set[str]:
        return {item.name for item in self.components if item.category == "solver_module"}

    @property
    def fv_models(self) -> set[str]:
        return {item.name for item in self.components if item.category == "fv_model"}

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
