"""Closed-region, saved-sample conservation checks with explicit equation scope."""
from __future__ import annotations
import math
from openfoam_agent.postprocessing.native_fields import read_mesh,read_field,patch_values,check_sources_current

RATE_DIM={"mass":(1,0,-1,0,0,0,0),"energy":(1,2,-3,0,0,0,0),"volume":(0,3,-1,0,0,0,0)}
RATE_UNIT={"mass":"kg/s","energy":"W","volume":"m3/s"}


def analyze_conservation(workspace,check,plan):
    from openfoam_agent.contracts.regions import region_layouts
    known={item.region for item in region_layouts(plan)}
    if any(item.region not in known for item in check.regions):raise ValueError("Conservation references an undeclared region.")
    rate=RATE_DIM[check.kind];storage=list(rate);storage[1]-=3;storage[2]+=1
    source=list(rate);source[1]-=3
    evidence=[];observations={};intervals=[];interfaces=[]
    for region in check.regions:
        mesh=read_mesh(workspace,region.region,evidence);samples=[]
        for name in check.time_names:
            flux=read_field(workspace,name,region.region,region.flux_field,mesh,evidence,expected_dimensions=rate,expected_class="surfaceScalarField")
            patches={};absolute_flux=0.0
            for patch in mesh.patches:
                values=patch_values(flux,mesh,patch);patches[patch]=sum(values);absolute_flux+=sum(abs(x) for x in values)
            stored=generated=0.0
            if region.storage_density_field:
                f=read_field(workspace,name,region.region,region.storage_density_field,mesh,evidence,expected_dimensions=storage,expected_class="volScalarField")
                stored=sum(v*vol for v,vol in zip(f["internal"],mesh.volumes))
            if region.source_density_field:
                f=read_field(workspace,name,region.region,region.source_density_field,mesh,evidence,expected_dimensions=source,expected_class="volScalarField")
                generated=sum(v*vol for v,vol in zip(f["internal"],mesh.volumes))
            samples.append({"time":float(name),"storage":stored,"source":generated,"outward_flux":sum(patches.values()),"absolute_flux":absolute_flux,"patches":patches})
        observations[region.region]=samples
        for a,b in zip(samples,samples[1:]):
            dt=b["time"]-a["time"];change=(b["storage"]-a["storage"])/dt
            flux=(a["outward_flux"]+b["outward_flux"])/2;src=(a["source"]+b["source"])/2
            residual=change+flux-src
            scale=abs(change)+(a["absolute_flux"]+b["absolute_flux"])/2+(abs(a["source"])+abs(b["source"]))/2
            allowed=check.absolute_tolerance+check.relative_tolerance*scale
            if any(not math.isfinite(x) for x in (change,flux,src,residual,scale,allowed)):raise ValueError("Nonfinite conservation arithmetic.")
            intervals.append({"region":region.region,"start":a["time"],"end":b["time"],"storage_rate":change,"outward_flux":flux,"source_rate":src,"residual":residual,"tolerance":allowed,"passed":abs(residual)<=allowed})
    if check.check_interfaces:
        seen=set()
        for interface in plan.interfaces:
            a,b=interface.region,interface.neighbour_region
            if a not in observations and b not in observations:continue
            if a not in observations or b not in observations:raise ValueError("A requested interface check must include both adjacent regions.")
            key=tuple(sorted(((a,interface.patch),(b,interface.neighbour_patch))))
            if key in seen:continue
            seen.add(key)
            for x,y in zip(observations[a],observations[b]):
                if interface.patch not in x["patches"] or interface.neighbour_patch not in y["patches"]:raise ValueError("Declared interface patch is absent in native evidence.")
                q1,q2=x["patches"][interface.patch],y["patches"][interface.neighbour_patch]
                residual=q1+q2;allowed=check.absolute_tolerance+check.relative_tolerance*(abs(q1)+abs(q2))
                interfaces.append({"pair":key,"time":x["time"],"residual":residual,"tolerance":allowed,"passed":abs(residual)<=allowed})
    passed=all(x["passed"] for x in intervals+interfaces)
    check_sources_current(workspace,evidence)
    return {"id":check.id,"kind":check.kind,"rate_unit":RATE_UNIT[check.kind],"arithmetic_verified":True,
        "physical_semantics_verified":True,"conservation_verified":passed,"sources":evidence,"intervals":intervals,"interfaces":interfaces,
        "equation":"d(storage)/dt + sum(outward boundary flux) - integrated source = 0",
        "equation_contract":check.model_dump(mode="json"),"governing_equation_completeness_verified":False,
        "limitations":["Conservation is checked for the explicitly approved equation on saved samples, not for all unsaved solver steps.",
            "Steady storage and zero source are explicit approved assumptions; automatic discovery of every physical model source is not claimed.",
            "Rates use trapezoidal interpolation; storage uses saved endpoint differences. Static reconstructed ASCII/gzip fields only."]}
