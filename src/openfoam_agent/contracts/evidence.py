"""Implementation evidence is data with provenance, not remembered chat history."""
from __future__ import annotations
import hashlib
from pathlib import PurePosixPath
from openfoam_agent.llm.context import ContextBudgetError


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def implementation_evidence_pack(state, plan, *, max_chars: int = 16000):
    from openfoam_agent.schemas.engineering import canonical_engineering_evidence_id
    available = {}
    for record in state.engineering_evidence_records:
        for item in _walk(record.payload):
            reference = item.get("reference")
            body = item.get("content") or item.get("content_excerpt")
            # Search snippets are discovery, not an observed syntax read.
            if not isinstance(reference, str) or not isinstance(body, str) or not body.strip():
                continue
            eid = canonical_engineering_evidence_id("openfoam_reference", reference)
            previous = available.get(eid)
            if previous is None or len(body) > len(previous["content"]):
                available[eid] = {
                    "evidence_id": eid, "source": reference, "record_id": record.record_id,
                    "content": body, "excerpt_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "is_excerpt": True, "openfoam_version": plan.openfoam_version,
                }
    bindings = {x.path: x.evidence_ids for x in plan.implementation_evidence_bindings}
    selected = {}
    coverage = []
    for path in plan.required_case_files:
        ids = bindings.get(path, [])
        if not ids:
            # Conservative inferred projection; a name match is explicitly not a
            # certified syntax/compatibility claim.
            basename = PurePosixPath(path).name
            ids = [eid for eid, item in available.items()
                   if basename in item["source"] or basename in item["content"]]
        missing = [eid for eid in ids if eid not in available]
        if missing:
            raise ValueError(f"Implementation evidence for {path} was not actually observed: {missing}")
        for eid in ids:
            selected[eid] = available[eid]
        coverage.append({"path": path, "evidence_ids": ids,
                         "status": "explicit" if bindings.get(path) else "inferred" if ids else "missing"})
    # Keep explicitly referenced implementation evidence even for auxiliary files.
    for path, ids in bindings.items():
        if path not in plan.required_case_files:
            raise ValueError(f"Implementation evidence binding references undeclared file: {path}")
    total = sum(len(item["content"]) for item in selected.values())
    if total > max_chars:
        raise ContextBudgetError("Mandatory implementation evidence exceeds the authoring budget; split the design into file/region bundles. No syntax evidence was truncated.")
    return {"records": list(selected.values()), "file_coverage": coverage,
            "complete": all(x["status"] == "explicit" for x in coverage),
            "scope": "Observed excerpts only; not proof of native compatibility."}


def evidence_coverage_failures(plan, state):
    pack = implementation_evidence_pack(state, plan)
    return [f"Explicit syntax evidence binding missing for {item['path']}."
            for item in pack["file_coverage"] if item["status"] != "explicit"]
