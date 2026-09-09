from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openfoam_agent.schemas.intake import IntakeFact


AssuranceMode = Literal[
    "routing_provenance",
    "artifact_recommended",
    "numeric_relation_recommended",
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

    # A direct single numeric user requirement can often be recomputed from one or
    # more artifacts.  When the Agent supplies such a relation Python verifies it
    # strictly, but not every quantity has a generic file representation.
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
