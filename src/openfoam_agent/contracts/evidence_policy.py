"""Risk- and stage-aware evidence policy for autonomous engineering.

Evidence is not a universal precondition.  Critical execution/safety facts remain
fail-closed, implementation syntax may be deferred until authoring, and ordinary
engineering judgement may proceed with explicit provenance and later validation.
"""
from __future__ import annotations

from typing import Literal

EvidencePolicy = Literal["mandatory", "deferred", "advisory"]
RiskLevel = Literal["low", "medium", "high", "critical"]
VerificationStage = Literal[
    "design", "authoring", "pre_validation", "pre_execution", "runtime", "result_review"
]

POLICY_SUMMARY = {
    "mandatory": {
        "meaning": "absence blocks at the named verification stage",
        "examples": [
            "confirmed user requirements",
            "trusted installed execution provider",
            "execution approval",
            "current mesh/checkMesh freshness",
            "case seal and completion evidence",
        ],
    },
    "deferred": {
        "meaning": "design may proceed; resolve or replace with deterministic/native validation before use",
        "examples": [
            "version-specific dictionary syntax",
            "specialized boundary/model implementation details",
            "region-specific file layout details",
        ],
    },
    "advisory": {
        "meaning": "absence never blocks by itself; record an engineering assumption/default and validate outcomes",
        "examples": [
            "initial mesh resolution",
            "representative timestep",
            "ordinary numerical controls",
            "engineering strategy preference",
        ],
    },
}

_EXECUTABLE_LEVELS = {"binary_present", "runtime_registered", "runtime_tested", "numerically_validated"}
_COMPONENT_LEVELS = {
    "installed", "source_discovered", "binary_present", "runtime_registered",
    "runtime_tested", "numerically_validated",
}

def provider_is_sufficient(provider, *, executable: bool) -> bool:
    """Whether deterministic catalog evidence is strong enough at design stage.

    This does not claim native runtime success.  Executables must have binary-or-stronger
    evidence; runtime-selectable components may be source/installed discovered because
    later authoring/pre-solve/native validation remains authoritative.
    """
    if provider is None or not bool(getattr(provider, "verified", False)):
        return False
    level = str(getattr(provider, "verification_level", "unverified"))
    return level in (_EXECUTABLE_LEVELS if executable else _COMPONENT_LEVELS)
