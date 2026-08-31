from __future__ import annotations

import re
from pathlib import Path

from openfoam_agent.capabilities.graph import CapabilityGraph


class CapabilityCatalog:
    """Read-only capability evidence for CFDEngineeringAgent.

    It deliberately has no coverage threshold, ranking policy, gap planner, or
    solver-selection method. Search scores are only lexical retrieval scores.
    """

    def __init__(self, graph_path: str | Path):
        self.graph = CapabilityGraph.from_json(graph_path)

    def summary(self) -> dict[str, object]:
        return {
            "graph_id": self.graph.spec.graph_id,
            "fingerprint": self.graph.fingerprint,
            "openfoam_distribution": self.graph.spec.openfoam_distribution,
            "openfoam_version": self.graph.spec.openfoam_version,
            "provider_count": len(self.graph.spec.providers),
        }

    def provider(self, provider_id: str):
        for provider in self.graph.spec.providers:
            if provider.id == provider_id:
                return provider
        return None

    def search(self, query: str, *, limit: int = 12) -> list[dict[str, object]]:
        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9_.-]+", query.casefold())
            if len(token) > 1
        ]
        candidates: list[tuple[int, str, dict[str, object]]] = []
        for provider in self.graph.spec.providers:
            haystack = " ".join(
                [provider.id, provider.name, provider.provider_type, *provider.capabilities]
            ).casefold()
            score = sum(
                3 if token in provider.id.casefold() else 1
                for token in tokens
                if token in haystack
            )
            if not tokens:
                score = 1
            if score <= 0:
                continue
            candidates.append(
                (
                    score,
                    provider.id,
                    {
                        "provider_id": provider.id,
                        "name": provider.name,
                        "provider_type": provider.provider_type,
                        "openfoam_version": provider.openfoam_version,
                        "capabilities": list(provider.capabilities),
                        "verified": provider.verified,
                        "verification_level": provider.verification_level,
                        "evidence": [item.model_dump(mode="json") for item in provider.evidence],
                        "extension_points": list(provider.extension_points),
                    },
                )
            )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in candidates[:limit]]
