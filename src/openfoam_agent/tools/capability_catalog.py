from __future__ import annotations

import re
from pathlib import Path

from openfoam_agent.capabilities.graph import CapabilityGraph
from openfoam_agent.schemas.capability import CapabilityEvidence, CapabilityProvider
from openfoam_agent.schemas.installation import InstalledOpenFOAMIR
from .references import normalize_query


class CapabilityCatalog:
    """Read-only documented + installed capability evidence for CFDEngineeringAgent.

    Static v13/v14 graphs supply documented semantics. The sourced installation is
    authoritative for executable availability and augments the graph with every trusted
    application plus runtime-selectable components actually discovered from the installed source tree.
    """

    def __init__(self, graph_path: str | Path, *, installation: InstalledOpenFOAMIR | None = None):
        self.graph = CapabilityGraph.from_json(graph_path)
        self.installation = installation
        self._installed = self._installed_providers(installation) if installation is not None else []

    def summary(self) -> dict[str, object]:
        return {
            "graph_id": self.graph.spec.graph_id,
            "fingerprint": self.graph.fingerprint,
            "openfoam_distribution": self.graph.spec.openfoam_distribution,
            "openfoam_version": self.graph.spec.openfoam_version,
            "provider_count": len(self.graph.spec.providers),
            "installed_provider_count": len(self._installed),
            "installed_ir_fingerprint": self.installation.fingerprint if self.installation is not None else None,
        }

    def all_providers(self) -> list[CapabilityProvider]:
        """Merge documented semantics with independently observed installation presence."""
        installed = list(self._installed)
        used: set[int] = set()
        merged: dict[str, CapabilityProvider] = {}
        for provider in self.graph.spec.providers:
            match_index = next((
                index for index, observed in enumerate(installed)
                if index not in used
                and observed.name == provider.name
                and _provider_family(observed.provider_type) == _provider_family(provider.provider_type)
            ), None)
            if match_index is None:
                metadata = dict(provider.metadata)
                metadata.setdefault("semantic_authority_documented_graph", True)
                merged[provider.id] = provider.model_copy(update={"metadata": metadata})
                continue
            observed = installed[match_index]
            used.add(match_index)
            evidence = list(provider.evidence)
            for item in observed.evidence:
                if item not in evidence:
                    evidence.append(item)
            metadata = dict(provider.metadata)
            metadata.update(observed.metadata)
            metadata["semantic_authority_documented_graph"] = True
            metadata["installed_observed"] = True
            merged[provider.id] = provider.model_copy(update={
                "verified": True,
                "verification_level": observed.verification_level,
                "evidence": evidence,
                "metadata": metadata,
            })
        for index, provider in enumerate(installed):
            if index not in used:
                merged[provider.id] = provider
        return [merged[key] for key in sorted(merged)]

    def provider(self, provider_id: str):
        for provider in self.all_providers():
            if provider.id == provider_id:
                return provider
        return None

    def search(self, query: str, *, limit: int = 12) -> list[dict[str, object]]:
        tokens = normalize_query(query)
        candidates: list[tuple[int, str, dict[str, object]]] = []
        for provider in self.all_providers():
            haystack = " ".join(
                [provider.id, provider.name, provider.provider_type, *provider.capabilities]
            ).casefold()
            score = sum(
                3 if token in provider.id.casefold() else 1
                for token in tokens
                if token in haystack
            )
            if not tokens:
                continue
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
                        "readiness_metadata": provider.metadata,
                        "evidence": [item.model_dump(mode="json") for item in provider.evidence],
                        "extension_points": list(provider.extension_points),
                    },
                )
            )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in candidates[:limit]]

    @staticmethod
    def _installed_providers(installation: InstalledOpenFOAMIR) -> list[CapabilityProvider]:
        if installation.version is None:
            return []
        evidence = [CapabilityEvidence(
            kind="installation_discovery",
            reference=f"installed:foundation:{installation.version}",
            note="Observed from the sourced trusted OpenFOAM installation; proves identity/presence only.",
        )]
        providers: list[CapabilityProvider] = []
        for item in installation.executables:
            if item.category == "execution_driver":
                provider_type = "execution_driver"
                capabilities = [f"execution.driver.{item.name}"]
            elif item.category == "solver_application":
                provider_type = "solver_application"
                capabilities = [f"solver.application.{item.name}"]
            else:
                provider_type = "utility"
                capabilities = [f"application.{item.name}"]
            providers.append(CapabilityProvider(
                id=f"installed.application.{item.name}", name=item.name,
                provider_type=provider_type, capabilities=capabilities,
                openfoam_version=installation.version, verified=True,
                verification_level="binary_present", evidence=evidence,
                metadata={
                    "runtime_load_verified": False, "native_test_verified": False,
                    "semantic_authority_documented_graph": False,
                    "semantic_capabilities_inferred": False,
                },
            ))
        for item in installation.components:
            if item.category == "solver_module":
                provider_type, provider_id, capabilities = "solver_module", f"installed.solver_module.{item.name}", [f"solver.module.{item.name}"]
            elif item.category == "fv_model":
                provider_type, provider_id, capabilities = "fv_model", f"installed.fv_model.{item.name}", [f"fvModel.{item.name}"]
            elif item.category == "function_object":
                provider_type, provider_id, capabilities = "function_object", f"installed.function_object.{item.name}", [f"functionObject.{item.name}"]
            elif item.category == "source_component":
                provider_type, provider_id, capabilities = "source_component", f"installed.source_component.{item.name}", [f"source.component.{item.name}"]
            else:
                continue
            providers.append(CapabilityProvider(
                id=provider_id, name=item.name, provider_type=provider_type, capabilities=capabilities,
                openfoam_version=installation.version, verified=True, verification_level="source_discovered",
                evidence=evidence, metadata={
                    "runtime_load_verified": False, "native_test_verified": False,
                    "semantic_authority_documented_graph": False,
                    "semantic_capabilities_inferred": False,
                },
            ))
        return providers


def _provider_family(provider_type: str) -> str:
    if provider_type in {"solver", "solver_module", "generated_solver"}:
        return "solver"
    return provider_type
