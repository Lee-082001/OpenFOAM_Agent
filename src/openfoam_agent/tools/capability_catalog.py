from __future__ import annotations

from pathlib import Path

from openfoam_agent.capabilities.graph import CapabilityGraph
from openfoam_agent.schemas.capability import CapabilityEvidence, CapabilityProvider
from openfoam_agent.schemas.installation import InstalledOpenFOAMIR
from .references import normalize_query


class CapabilityCatalog:
    """Read-only documented + installed capability evidence for CFDEngineeringAgent.

    The capability graph owns documented CFD semantics. Installation discovery owns only
    existence/readiness observations.  Python never infers physics capability from an
    executable/component name.  When a documented provider and an installed identity
    observation describe the same provider, the catalog joins those independent facts
    without manufacturing additional semantics.
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

    @staticmethod
    def _types_compatible(documented: str, installed: str) -> bool:
        if documented == installed:
            return True
        # Historical graphs use both ``solver`` and ``solver_module`` for modular
        # Foundation solvers.  Treat that naming difference as identity compatibility,
        # not as a semantic capability inference.
        return {documented, installed} <= {"solver", "solver_module"}

    def _installed_match(self, provider: CapabilityProvider) -> CapabilityProvider | None:
        for observed in self._installed:
            if observed.name != provider.name:
                continue
            if self._types_compatible(provider.provider_type, observed.provider_type):
                return observed
        return None

    def all_providers(self) -> list[CapabilityProvider]:
        merged: dict[str, CapabilityProvider] = {}
        for provider in self.graph.spec.providers:
            observed = self._installed_match(provider)
            if observed is None:
                merged[provider.id] = provider
                continue
            metadata = dict(provider.metadata)
            metadata.update(
                installed_observed=True,
                installed_provider_id=observed.id,
                installed_verification_level=observed.verification_level,
                semantic_capabilities_inferred=False,
            )
            evidence = list(provider.evidence)
            for item in observed.evidence:
                if item not in evidence:
                    evidence.append(item)
            # Keep the graph's documented verification level: documentation proves the
            # semantics while the joined installation observation proves local presence.
            merged[provider.id] = provider.model_copy(
                update={"metadata": metadata, "evidence": evidence, "verified": True}
            )
        for item in self._installed:
            merged.setdefault(item.id, item)
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
        evidence = [
            CapabilityEvidence(
                kind="installation_discovery",
                reference=f"installed:foundation:{installation.version}",
                note=(
                    "Observed in the sourced trusted OpenFOAM installation. This evidence "
                    "proves local identity/presence only; no CFD semantics are inferred from the name."
                ),
            )
        ]
        common_metadata = {
            "runtime_load_verified": False,
            "native_test_verified": False,
            "semantic_capabilities_inferred": False,
        }
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
            providers.append(
                CapabilityProvider(
                    id=f"installed.application.{item.name}",
                    name=item.name,
                    provider_type=provider_type,
                    capabilities=capabilities,
                    openfoam_version=installation.version,
                    verified=True,
                    verification_level="binary_present",
                    metadata=dict(common_metadata),
                    evidence=evidence,
                )
            )
        for item in installation.components:
            if item.category == "solver_module":
                ptype = "solver_module"
                capabilities = [f"solver.module.{item.name}"]
                provider_id = f"installed.solver_module.{item.name}"
            elif item.category == "fv_model":
                ptype = "fv_model"
                capabilities = [f"fvModel.{item.name}"]
                provider_id = f"installed.fv_model.{item.name}"
            elif item.category == "function_object":
                ptype = "function_object"
                capabilities = [f"functionObject.{item.name}"]
                provider_id = f"installed.function_object.{item.name}"
            elif item.category == "source_component":
                ptype = "source_component"
                capabilities = [f"source.component.{item.name}"]
                provider_id = f"installed.source_component.{item.name}"
            else:
                continue
            providers.append(
                CapabilityProvider(
                    id=provider_id,
                    name=item.name,
                    provider_type=ptype,
                    capabilities=capabilities,
                    openfoam_version=installation.version,
                    verified=True,
                    verification_level="source_discovered",
                    metadata=dict(common_metadata),
                    evidence=evidence,
                )
            )
        return providers
