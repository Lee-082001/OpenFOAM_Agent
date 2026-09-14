from __future__ import annotations

from pathlib import Path

from openfoam_agent.capabilities.graph import CapabilityGraph
from openfoam_agent.schemas.capability import CapabilityEvidence, CapabilityProvider
from openfoam_agent.schemas.installation import InstalledOpenFOAMIR
from .references import normalize_query


class CapabilityCatalog:
    """Read-only documented + installed capability evidence for CFDEngineeringAgent.

    The capability graph owns semantic claims. Installation discovery owns only
    existence/registration observations. A provider receives design-stage installed
    verification only when a documented semantic provider can be joined to a matching
    installed identity by name/version/type compatibility. Python never manufactures
    physics capability strings from a component name.
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
    def _provider_types_compatible(documented: str, installed: str) -> bool:
        if documented == installed:
            return True
        # Foundation capability graphs historically used `solver` for runtime
        # selectable modules while installation discovery can observe them as
        # `solver_module`. This is a representation compatibility rule only; it
        # does not infer any physics capability.
        return {documented, installed} <= {"solver", "solver_module"}

    def _joined_documented_provider(self, provider: CapabilityProvider) -> CapabilityProvider:
        matches = [
            item
            for item in self._installed
            if item.name == provider.name
            and item.openfoam_version == provider.openfoam_version
            and self._provider_types_compatible(provider.provider_type, item.provider_type)
        ]
        if not matches:
            return provider
        installed = sorted(matches, key=lambda item: item.id)[0]
        evidence = [*provider.evidence]
        for item in installed.evidence:
            if item.model_dump(mode="json") not in [x.model_dump(mode="json") for x in evidence]:
                evidence.append(item)
        metadata = dict(provider.metadata)
        metadata.update(
            {
                "installed_identity_provider": installed.id,
                "semantic_capability_source": provider.id,
                "runtime_load_verified": bool(installed.metadata.get("runtime_load_verified", False)),
                "native_test_verified": bool(installed.metadata.get("native_test_verified", False)),
            }
        )
        return provider.model_copy(
            update={
                "verified": bool(provider.verified and installed.verified),
                "verification_level": installed.verification_level,
                "evidence": evidence,
                "metadata": metadata,
            }
        )

    def all_providers(self) -> list[CapabilityProvider]:
        documented = [self._joined_documented_provider(item) for item in self.graph.spec.providers]
        merged: dict[str, CapabilityProvider] = {item.id: item for item in documented}

        # Keep unmatched installation identities queryable, but with identity-only
        # capability strings. These records prove presence/registration, not CFD
        # semantics. Documented providers above remain the semantic authority.
        documented_matches = {
            (item.name, item.openfoam_version, item.provider_type)
            for item in documented
        }
        for item in self._installed:
            matched = any(
                name == item.name
                and version == item.openfoam_version
                and self._provider_types_compatible(ptype, item.provider_type)
                for name, version, ptype in documented_matches
            )
            if not matched:
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
                note="Observed in the sourced trusted OpenFOAM installation; this evidence proves identity/presence only.",
            )
        ]
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
                    metadata={"runtime_load_verified": False, "native_test_verified": False},
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
                    metadata={"runtime_load_verified": False, "native_test_verified": False},
                    evidence=evidence,
                )
            )
        return providers
