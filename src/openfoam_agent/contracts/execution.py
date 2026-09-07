from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any
from pydantic import Field
from .models import Contract, ResourceLimits

if TYPE_CHECKING:
    from openfoam_agent.schemas.engineering import EngineeringPlan, CaseSeal


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def execution_document(plan: EngineeringPlan) -> dict[str, Any]:
    if plan.execution is not None:
        return plan.execution.model_dump(mode="json")
    return {"driver": "foamRun", "solver_module": plan.solver, "solver_provider_id": plan.solver_provider_id,
            "arguments": [], "regions": [], "parallel": {"mode": "serial", "ranks": 1}}


def physical_plan_document(plan: EngineeringPlan) -> dict[str, Any]:
    return {key: plan.model_dump(mode="json")[key] for key in (
        "confirmed_intake_sha256", "confirmed_fact_ids", "confirmed_fact_bindings", "temporal_behavior",
        "motion_kind", "mesh_motion_requirement", "required_case_files", "region_layouts", "interfaces", "completion", "quantities_of_interest", "conservation_checks",
    )}


class ExecutionApproval(Contract):
    execution: dict[str, Any]
    execution_sha256: str
    physical_plan_sha256: str
    intake_sha256: str
    approved_manifest_sha256: str
    approved_files: dict[str, str] = Field(default_factory=dict)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    approved_by: str = "explicit_user_solve"

    @classmethod
    def issue(cls, plan: EngineeringPlan, seal: CaseSeal, limits: ResourceLimits | None = None) -> ExecutionApproval:
        execution = execution_document(plan)
        return cls(execution=execution, execution_sha256=digest(execution),
                   physical_plan_sha256=digest(physical_plan_document(plan)),
                   intake_sha256=plan.confirmed_intake_sha256,
                   approved_manifest_sha256=seal.manifest_sha256,
                   approved_files={x.path: x.sha256 for x in seal.files},
                   resource_limits=limits or ResourceLimits())

    def check_plan(self, plan: EngineeringPlan) -> None:
        if digest(self.execution) != self.execution_sha256:
            raise ValueError("Execution approval is internally inconsistent.")
        if digest(execution_document(plan)) != self.execution_sha256:
            raise ValueError("Execution topology/arguments/resources changed; explicit approval is required.")
        if digest(physical_plan_document(plan)) != self.physical_plan_sha256:
            raise ValueError("Confirmed physical implementation contract changed; explicit approval is required.")
        if plan.confirmed_intake_sha256 != self.intake_sha256:
            raise ValueError("Approved intake changed.")

    def check_repair_files(self, seal: CaseSeal) -> None:
        """Conservative automatic repair: numerical controls only, no silent physics edits."""
        current = {x.path: x.sha256 for x in seal.files}
        changed = {p for p in set(current) | set(self.approved_files) if current.get(p) != self.approved_files.get(p)}
        allowed = {p for p in changed if p.startswith("system/") and p.rsplit("/", 1)[-1] in {"fvSchemes", "fvSolution"}}
        if changed - allowed:
            raise ValueError("Repair changes inputs outside numerical-control authorization: " + ", ".join(sorted(changed - allowed)))
