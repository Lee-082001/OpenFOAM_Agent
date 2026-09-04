from __future__ import annotations

import re
from pathlib import Path

from openfoam_agent.capabilities.graph import CapabilityGraph
from openfoam_agent.schemas.capability import CapabilityEvidence, CapabilityProvider
from openfoam_agent.schemas.installation import InstalledOpenFOAMIR


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
        merged: dict[str, CapabilityProvider] = {item.id: item for item in self.graph.spec.providers}
        for item in self._installed:
            merged.setdefault(item.id, item)
        return [merged[key] for key in sorted(merged)]

    def provider(self, provider_id: str):
        for provider in self.all_providers():
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

    @staticmethod
    def _installed_providers(installation: InstalledOpenFOAMIR) -> list[CapabilityProvider]:
        if installation.version is None:
            return []
        evidence = [
            CapabilityEvidence(
                kind="installation_discovery",
                reference=f"installed:foundation:{installation.version}",
                note="Discovered from the sourced trusted OpenFOAM installation.",
            )
        ]
        providers: list[CapabilityProvider] = []
        for item in installation.executables:
            if item.category == "execution_driver":
                provider_type = "execution_driver"
                capabilities = [f"execution.driver.{item.name}"]
                if item.name == "foamMultiRun":
                    capabilities += ["execution.multiregion", "heat_transfer.conjugate"]
                elif item.name == "foamRun":
                    capabilities += ["execution.single_region", "execution.solver_module"]
            elif item.category == "solver_application":
                provider_type = "solver_application"
                capabilities = [f"solver.application.{item.name}"]
            else:
                provider_type = "utility"
                capabilities = [f"utility.{item.name}", f"application.{item.name}"]
            providers.append(
                CapabilityProvider(
                    id=f"installed.application.{item.name}",
                    name=item.name,
                    provider_type=provider_type,
                    capabilities=capabilities,
                    openfoam_version=installation.version,
                    verified=True,
                    verification_level="installed",
                    evidence=evidence,
                )
            )
        for item in installation.components:
            if item.category == "solver_module":
                ptype = "solver_module"
                capabilities = [f"solver.module.{item.name}"]
                if item.name == "solid":
                    capabilities += ["heat_transfer.solid", "equation.energy.solid", "heat_transfer.conjugate"]
                elif item.name == "fluid":
                    capabilities += ["heat_transfer.fluid", "equation.energy.temperature", "heat_transfer.conjugate"]
                elif item.name == "incompressibleFluid":
                    capabilities += ["flow.incompressible"]
                provider_id = f"installed.solver_module.{item.name}"
            elif item.category == "fv_model":
                ptype = "fv_model"
                capabilities = [f"fvModel.{item.name}"]
                if item.name == "heatSource":
                    capabilities += ["source.heat.volumetric", "heat_generation.volumetric"]
                elif item.name in {"solidificationMelting", "VoFSolidificationMelting"}:
                    capabilities += [
                        "phase_change.solid_liquid",
                        "melting",
                        "solidification",
                        "energy.latent_heat",
                    ]
                    if item.name == "VoFSolidificationMelting":
                        capabilities.append("multiphase.vof")
                    else:
                        capabilities.append("phase_change.enthalpy_porosity")
                elif item.name in {"heatTransferLimitedPhaseChange", "coefficientPhaseChange"}:
                    capabilities += [
                        "phase_change.fluid_fluid",
                        "phase_change.mass_transfer",
                    ]
                provider_id = f"installed.fv_model.{item.name}"
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
                    verification_level="installed",
                    evidence=evidence,
                )
            )
        return providers
