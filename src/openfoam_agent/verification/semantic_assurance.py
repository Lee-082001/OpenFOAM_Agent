from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from openfoam_agent.schemas.intake import IntakeFact



_NUMERIC = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.])")


def _is_critical_quantitative_user_fact(fact: IntakeFact) -> bool:
    if fact.source != "user":
        return False
    if fact.quantity is not None:
        return True
    if len(_NUMERIC.findall(fact.value)) != 1:
        return False
    if fact.category in {"scale", "property"}:
        return True
    return bool(fact.unit) and fact.category in {
        "geometry", "material", "physics", "temporal", "motion", "boundary"
    }

AssuranceMode = Literal[
    "routing_provenance",
    "artifact_recommended",
    "numeric_relation_recommended",
    "machine_assertion_required",
    "provenance",
]


@dataclass(frozen=True)
class SemanticAssuranceExpectation:
    """What deterministic assurance is reasonable for one confirmed intake fact.

    The key distinction is between *preservation* and *independent machine proof*.
    Every non-context fact must remain provenance-bound to the frozen intake.  Only
    facts with a reliable artifact representation are candidates for an additional
    machine assertion.  Absence of that optional assertion is an assurance gap, not
    evidence that the CFD case is wrong.
    """

    mode: AssuranceMode
    machine_assertion_recommended: bool
    reason: str


def expectation_for_fact(fact: IntakeFact) -> SemanticAssuranceExpectation:
    # These values guide problem routing, interpretation or requested observations;
    # they are not direct OpenFOAM dictionary values.  Requiring a fabricated
    # snippet for them creates false failures such as classification.internal_flow.
    if fact.category in {"context", "classification", "objective", "assumption"}:
        return SemanticAssuranceExpectation(
            "routing_provenance",
            False,
            "Routing/interpretation fact is preserved by intake digest and binding provenance, not by an invented case-file token.",
        )

    # v5.1 reliability contract: an explicitly typed physical quantity supplied by
    # the user is no longer provenance-only.  It must be tied to the selected case
    # by either an exact dictionary/content assertion or a numeric recomputation.
    # The quantity object is created only after intake-side unit/dimension checks, so
    # this rule does not guess that arbitrary digits in natural language are physics.
    if _is_critical_quantitative_user_fact(fact):
        return SemanticAssuranceExpectation(
            "machine_assertion_required",
            True,
            "Explicit user physical quantities require machine-verifiable implementation evidence in the authored case.",
        )

    # Untyped direct numeric facts retain the v5.0 advisory policy because counts,
    # labels and strategy-dependent quantities do not all have a universal artifact
    # representation.
    if fact.source == "user" and fact.category in {"scale", "property", "physics"}:
        return SemanticAssuranceExpectation(
            "numeric_relation_recommended",
            True,
            "Direct numeric user requirements benefit from a machine-checkable relation when a stable artifact mapping exists.",
        )

    # These categories commonly affect concrete dictionaries/fields/geometry, but
    # the exact OpenFOAM representation is engineering-strategy dependent.  A
    # supplied assertion is therefore verified strictly; its absence is advisory.
    if fact.category in {
        "domain",
        "geometry",
        "material",
        "physics",
        "temporal",
        "motion",
        "boundary",
        "output",
        "fidelity",
    }:
        return SemanticAssuranceExpectation(
            "artifact_recommended",
            True,
            "The fact normally influences solver artifacts, but no universal file token exists across valid OpenFOAM strategies.",
        )

    return SemanticAssuranceExpectation(
        "provenance",
        False,
        "Confirmed fact is preserved through immutable intake provenance and its EngineeringPlan binding.",
    )
