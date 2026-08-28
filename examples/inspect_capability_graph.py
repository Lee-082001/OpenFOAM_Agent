from __future__ import annotations

import json
from pathlib import Path

from openfoam_agent.capabilities import CapabilityGraph


ROOT = Path(__file__).resolve().parents[1]
graph = CapabilityGraph.from_json(
    ROOT / "config" / "openfoam14_capability_graph.json"
)

summary = {
    "graph_id": graph.spec.graph_id,
    "schema_version": graph.spec.schema_version,
    "openfoam_distribution": graph.spec.openfoam_distribution,
    "openfoam_version": graph.spec.openfoam_version,
    "fingerprint": graph.fingerprint,
    "providers": [
        {
            "id": provider.id,
            "name": provider.name,
            "type": provider.provider_type,
            "verification_level": provider.verification_level,
            "capability_count": len(provider.capabilities),
            "evidence": [item.reference for item in provider.evidence],
        }
        for provider in graph.spec.providers
    ],
}

print(json.dumps(summary, indent=2))
