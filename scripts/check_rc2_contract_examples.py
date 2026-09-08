"""Validate example JSON contracts only; never provision isolation or run CFD."""
from pathlib import Path
from openfoam_agent.contracts.models import QuantityOfInterest, ConservationCheck
from openfoam_agent.tools.linux_isolation import LinuxIsolationPolicy
root=Path(__file__).resolve().parents[1]/"examples"/"v4rc2"
for name,model in [("isolation.policy.example.json",LinuxIsolationPolicy),
                   ("native_temperature.quantity.json",QuantityOfInterest),
                   ("mass_conservation.check.json",ConservationCheck)]:
    model.model_validate_json((root/name).read_text())
    print(name+": schema valid (not runtime validated)")
