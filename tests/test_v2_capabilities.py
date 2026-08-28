from __future__ import annotations

from openfoam_agent.capabilities import CapabilityGraph
from openfoam_agent.tools.capability_catalog import CapabilityCatalog


def test_capability_graph_loads_as_evidence(graph_path):
    graph = CapabilityGraph.from_json(graph_path)
    provider = graph.provider("solver.incompressibleFluid")
    assert provider.name == "incompressibleFluid"
    assert provider.verified
    assert "flow.momentum.incompressible" in provider.capabilities


def test_capability_catalog_retrieves_but_does_not_choose(graph_path):
    catalog = CapabilityCatalog(graph_path)
    results = catalog.search("incompressible transient momentum")
    assert any(item["provider_id"] == "solver.incompressibleFluid" for item in results)
    assert not hasattr(catalog, "plan")
    assert not hasattr(catalog, "choose_solver")
    assert not hasattr(catalog, "gap")
