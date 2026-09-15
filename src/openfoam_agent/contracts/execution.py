from __future__ import annotations

import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Any, Mapping
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


def _strip_foam_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if quote is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1]); i += 2; continue
            if ch == quote:
                quote = None
            i += 1; continue
        if ch in {'"', "'"}:
            quote = ch; out.append(ch); i += 1; continue
        if ch == "/" and nxt == "/":
            while i < len(text) and text[i] != "\n": i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(len(text), i + 2); continue
        out.append(ch); i += 1
    return "".join(out)


def _matching_brace(text: str, start: int) -> int | None:
    depth = 0
    quote: str | None = None
    i = start
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\": i += 2; continue
            if ch == quote: quote = None
        elif ch in {'"', "'"}: quote = ch
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return i
        i += 1
    return None


def _top_level_entry_payloads(text: str) -> tuple[dict[str, str], bool]:
    """Bounded controlDict projection used only for repair authorization."""
    clean = _strip_foam_comments(text)
    result: dict[str, str] = {}
    complete = True
    i = 0
    key_re = re.compile(r'[A-Za-z_][A-Za-z0-9_.-]*')
    while i < len(clean):
        while i < len(clean) and clean[i].isspace(): i += 1
        if i >= len(clean): break
        if clean[i] == '#':
            return result, False
        match = key_re.match(clean, i)
        if match is None:
            return result, False
        key = match.group(0); i = match.end()
        while i < len(clean) and clean[i].isspace(): i += 1
        if key in result: return result, False
        if i < len(clean) and clean[i] == '{':
            end = _matching_brace(clean, i)
            if end is None: return result, False
            result[key] = clean[i:end + 1].strip()
            i = end + 1
            continue
        start = i
        round_depth = square_depth = 0
        quote: str | None = None
        while i < len(clean):
            ch = clean[i]
            if quote is not None:
                if ch == "\\": i += 2; continue
                if ch == quote: quote = None
            elif ch in {'"', "'"}: quote = ch
            elif ch == '(': round_depth += 1
            elif ch == ')': round_depth = max(0, round_depth - 1)
            elif ch == '[': square_depth += 1
            elif ch == ']': square_depth = max(0, square_depth - 1)
            elif ch == ';' and round_depth == square_depth == 0:
                break
            elif ch == '{':
                # A nested dictionary without a literal top-level key/value boundary
                # is not safe to reason about as a scalar mutation.
                complete = False
            i += 1
        if i >= len(clean): return result, False
        payload = clean[start:i].strip()
        if not payload: complete = False
        result[key] = payload
        i += 1
    return result, complete


def canonical_payload(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    # Ignore formatting outside strings, preserving quoted text exactly.
    return tuple(re.findall(r'"(?:\\.|[^"\\])*"|\S+', value))


def literal_entry_payload(text: str, entry_path: str) -> str | None:
    payload = text
    parts = entry_path.split(".")
    for index, part in enumerate(parts):
        entries, complete = _top_level_entry_payloads(payload)
        if not complete or part not in entries:
            return None
        payload = entries[part]
        if index != len(parts) - 1:
            if not (payload.startswith("{") and payload.endswith("}")):
                return None
            payload = payload[1:-1]
    if "#" in payload or "$" in payload:
        return None
    return payload


def positive_scalar(value: str | None, label: str) -> float:
    try:
        result = float(value) if value is not None else float("nan")
    except ValueError as exc:
        raise ValueError(f"{label} requires a positive finite scalar.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} requires a positive finite scalar.")
    return result


_RUNTIME_NUMERICAL_DICTIONARIES = {"fvSchemes", "fvSolution"}
_RUNTIME_CONTROLDICT_MUTABLE_ENTRIES = {
    "deltaT", "maxDeltaT", "adjustTimeStep", "maxCo", "maxAlphaCo"
}


def _runtime_repair_path_kind(path: str) -> str | None:
    if not path.startswith("system/"):
        return None
    name = path.rsplit("/", 1)[-1]
    if name in _RUNTIME_NUMERICAL_DICTIONARIES:
        return "numerical_dictionary"
    if path == "system/controlDict":
        return "control_dict"
    return None


def physical_plan_document(plan: EngineeringPlan) -> dict[str, Any]:
    return {key: plan.model_dump(mode="json")[key] for key in (
        "confirmed_intake_sha256", "confirmed_fact_ids", "confirmed_fact_bindings", "temporal_behavior",
        "motion_kind", "mesh_motion_requirement", "required_case_files", "region_layouts", "interfaces", "completion", "quantities_of_interest", "conservation_checks",
    )}


def _protected_repair_locations(plan: EngineeringPlan) -> tuple[dict[str, list[str]], list[str]]:
    """Project machine-verifiable requirement bindings into repair protections."""
    entries: dict[str, set[str]] = {}
    whole_files: set[str] = set()
    for binding in plan.confirmed_fact_bindings:
        for assertion in binding.case_assertions:
            if assertion.entry_path:
                entries.setdefault(assertion.path, set()).add(assertion.entry_path)
            else:
                whole_files.add(assertion.path)
        relation = binding.numeric_relation
        if relation is not None:
            for term in [*relation.numerator, *relation.denominator]:
                if term.entry_path:
                    entries.setdefault(term.path, set()).add(term.entry_path)
                else:
                    whole_files.add(term.path)
    return ({path: sorted(values) for path, values in entries.items()}, sorted(whole_files))


class ExecutionApproval(Contract):
    execution: dict[str, Any]
    execution_sha256: str
    physical_plan_sha256: str
    intake_sha256: str
    approved_manifest_sha256: str
    approved_files: dict[str, str] = Field(default_factory=dict)
    protected_repair_entries: dict[str, list[str]] = Field(default_factory=dict)
    protected_repair_files: list[str] = Field(default_factory=list)
    # Controller-issued hashes for fine-grained runtime deltas that were checked
    # before mutation. This prevents the coarse post-mutation seal check from
    # treating arbitrary controlDict edits as authorized.
    authorized_repair_hashes: dict[str, str] = Field(default_factory=dict)
    repair_baselines: dict[str, str] = Field(default_factory=dict)
    authorized_repair_base_hashes: dict[str, str] = Field(default_factory=dict)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    approved_by: str = "explicit_user_solve"

    @classmethod
    def issue(cls, plan: EngineeringPlan, seal: CaseSeal, limits: ResourceLimits | None = None) -> ExecutionApproval:
        execution = execution_document(plan)
        protected_entries, protected_files = _protected_repair_locations(plan)
        return cls(execution=execution, execution_sha256=digest(execution),
                   physical_plan_sha256=digest(physical_plan_document(plan)),
                   intake_sha256=plan.confirmed_intake_sha256,
                   approved_manifest_sha256=seal.manifest_sha256,
                   approved_files={x.path: x.sha256 for x in seal.files},
                   protected_repair_entries=protected_entries,
                   protected_repair_files=protected_files,
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

    def check_repair_delta(self, workspace, changed_files: Mapping[str, str]) -> None:
        """Validate the complete delta before updating authorization or mutating files.

        Hashes bind the previous approved bytes and the exact proposed bytes. Values
        are checked against the first approved baseline and the last accepted value.
        Old checkpoints without these hashes fail closed for already changed files.
        """
        hashes = dict(self.authorized_repair_hashes)
        baselines = dict(self.repair_baselines)
        base_hashes = dict(self.authorized_repair_base_hashes)
        for path, proposed in changed_files.items():
            kind = _runtime_repair_path_kind(path)
            if kind is None:
                raise ValueError(f"Repair changes input outside numerical-control authorization: {path}")
            try:
                current = workspace.read_text(path, max_chars=workspace.max_file_bytes + 1)
            except (FileNotFoundError, OSError, UnicodeError) as exc:
                raise ValueError(f"Cannot inspect approved repair input {path}: {exc}") from exc
            current_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
            proposed_hash = hashlib.sha256(proposed.encode("utf-8")).hexdigest()
            approved_hash = self.approved_files.get(path)
            # Rechecking the same uncommitted candidate is deliberately idempotent.
            rechecking = proposed_hash == hashes.get(path) and current_hash == base_hashes.get(path)
            if current_hash not in {approved_hash, hashes.get(path)} and not rechecking:
                raise ValueError(f"Repair baseline has unapproved changes: {path}")
            if proposed == current:
                continue
            if path in self.protected_repair_files:
                raise ValueError(f"Repair would modify a frozen requirement implementation file: {path}")
            protected = self.protected_repair_entries.get(path, [])
            if kind == "numerical_dictionary":
                for entry in protected:
                    before = literal_entry_payload(current, entry)
                    after = literal_entry_payload(proposed, entry)
                    if before is None or after is None:
                        raise ValueError(f"Cannot resolve frozen requirement entry {path}:{entry}")
                    if canonical_payload(before) != canonical_payload(after):
                        raise ValueError(f"Repair would modify frozen requirement entry {path}:{entry}")
            else:
                before, complete_before = _top_level_entry_payloads(current)
                after, complete_after = _top_level_entry_payloads(proposed)
                if not complete_before or not complete_after:
                    raise ValueError("controlDict repair authorization requires complete literal top-level entries.")
                changed = {key for key in set(before) | set(after)
                           if canonical_payload(before.get(key)) != canonical_payload(after.get(key))}
                frozen = {entry.split(".", 1)[0] for entry in protected}
                if changed & frozen:
                    raise ValueError("Repair would modify frozen user requirement controlDict entries: " + ", ".join(sorted(changed & frozen)))
                forbidden = changed - _RUNTIME_CONTROLDICT_MUTABLE_ENTRIES
                if forbidden:
                    raise ValueError("Repair changes unauthorized controlDict entries: " + ", ".join(sorted(forbidden)))
                baseline = baselines.get(path, current)
                original, complete_original = _top_level_entry_payloads(baseline)
                if not complete_original:
                    raise ValueError("Cannot resolve original numerical repair limits.")
                for entry in changed:
                    if entry == "adjustTimeStep":
                        if after.get(entry) not in {"true", "false", "yes", "no", "on", "off", "0", "1"}:
                            raise ValueError("adjustTimeStep requires a literal boolean.")
                        if after[entry] in {"true", "yes", "on", "1"}:
                            # Enabling adaptive steps must not implicitly authorize
                            # increases beyond the original fixed timestep.
                            cap = positive_scalar(after.get("maxDeltaT"), "maxDeltaT")
                            limit = positive_scalar(original.get("maxDeltaT", original.get("deltaT")), "initial timestep limit")
                            if cap > limit:
                                raise ValueError("Adaptive timestep cap exceeds original approval.")
                        continue
                    new_value = positive_scalar(after.get(entry), entry)
                    prior = before.get(entry)
                    initial = original.get(entry)
                    # Only maxDeltaT can be added using an existing deltaT limit.
                    if entry == "maxDeltaT":
                        prior = prior or before.get("deltaT")
                        initial = initial or original.get("deltaT")
                    previous = positive_scalar(prior, f"previous {entry}")
                    initial_value = positive_scalar(initial, f"initial {entry}")
                    if new_value > min(previous, initial_value):
                        raise ValueError(f"Automatic repair may only reduce {entry} within original approval.")
                baselines.setdefault(path, current)
            base_hashes[path] = current_hash
            hashes[path] = proposed_hash
        # A rejected multi-file candidate must not leave partial authorization.
        self.authorized_repair_hashes = hashes
        self.repair_baselines = baselines
        self.authorized_repair_base_hashes = base_hashes

    def check_repair_files(self, seal: CaseSeal) -> None:
        """Post-mutation seal check; fine-grained authorization happens pre-mutation."""
        current = {x.path: x.sha256 for x in seal.files}
        changed = {p for p in set(current) | set(self.approved_files) if current.get(p) != self.approved_files.get(p)}
        allowed: set[str] = set()
        for path in changed:
            kind = _runtime_repair_path_kind(path)
            if kind in {"numerical_dictionary", "control_dict"} and current.get(path) is not None and self.authorized_repair_hashes.get(path) == current.get(path):
                allowed.add(path)
        if changed - allowed:
            raise ValueError("Repair changes inputs outside numerical-control authorization: " + ", ".join(sorted(changed - allowed)))
