"""Resolve a module/region without confusing a solver driver with a module."""
from openfoam_agent.contracts.regions import region_layouts


def resolve_postprocess_context(plan, region="", use_solver_context=True):
    layouts = region_layouts(plan, plan.required_case_files)
    regions = {layout.region: layout for layout in layouts}
    if region not in regions:
        raise ValueError("Post-processing must select one declared region explicitly.")
    if not use_solver_context:
        return None
    execution = plan.execution
    if execution is None:
        if plan.solver in {"foamRun", "foamMultiRun"}:
            raise ValueError("A driver name is not a post-processing solver module.")
        return plan.solver
    if execution.driver == "foamRun":
        return execution.solver_module
    if execution.driver == "foamMultiRun":
        return next(x.solver_module for x in execution.regions if x.region == region)
    # Direct applications have no implied modular post-processing counterpart.
    module = regions[region].solver_module
    if not module:
        raise ValueError("Direct application requires an explicit region solver_module for contextual post-processing, or use_solver_context=false.")
    return module
