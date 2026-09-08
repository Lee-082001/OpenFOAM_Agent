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


def observed_syntax(state, version="unknown"):
    """Keep distinct read windows, never silently replace one with a longer one."""
    from openfoam_agent.schemas.engineering import canonical_engineering_evidence_id
    available, targets = {}, {}
    for record in state.engineering_evidence_records:
        for item in _walk(record.payload):
            reference = item.get("reference")
            body = item.get("content") or item.get("content_excerpt")
            if not isinstance(reference, str) or not isinstance(body, str) or not body.strip():
                continue
            eid = canonical_engineering_evidence_id("openfoam_reference", reference)
            for target in item.get("target_case_files", []):
                targets.setdefault(target, []).append(eid)
            digest = hashlib.sha256(body.encode()).hexdigest()
            entry = available.setdefault(eid, {"evidence_id":eid, "source":reference,
                "record_id":record.record_id, "is_excerpt":True,
                "openfoam_version":version, "observed_windows":[]})
            if not any(w["excerpt_sha256"] == digest for w in entry["observed_windows"]):
                entry["observed_windows"].append({"record_id":record.record_id,
                    "content":body, "excerpt_sha256":digest})
    for entry in available.values():
        # Do not imply disjoint windows are a single continuous source excerpt.
        entry["content"] = "\n// --- separate observed read window ---\n".join(
            w["content"] for w in entry["observed_windows"])
        entry["excerpt_sha256"] = hashlib.sha256(entry["content"].encode()).hexdigest()
        # Avoid duplicating large bodies in every prompt; each window retains its
        # own record id, hash and character range into the aggregate content.
        offset=0
        for w in entry["observed_windows"]:
            body=w.pop("content");w.update(start_char=offset,end_char=offset+len(body))
            offset+=len(body)+len("\n// --- separate observed read window ---\n")
    return available, targets


def implementation_evidence_pack(state, plan, *, max_chars: int | None = 16000):
    available, targets = observed_syntax(state, plan.openfoam_version)
    bindings = {path: list(dict.fromkeys(ids)) for path,ids in targets.items() if path in plan.required_case_files}
    bindings.update({x.path: x.evidence_ids for x in plan.implementation_evidence_bindings if x.evidence_ids})
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
    if max_chars is not None and total > max_chars:
        raise ContextBudgetError("Mandatory implementation evidence exceeds the authoring budget; split the design into file/region bundles. No syntax evidence was truncated.")
    return {"records": list(selected.values()), "file_coverage": coverage,
            "complete": all(x["status"] == "explicit" for x in coverage),
            "scope": "Observed excerpts only; not proof of native compatibility."}


def evidence_coverage_failures(plan, state):
    pack = implementation_evidence_pack(state, plan, max_chars=None)
    return [f"Explicit syntax evidence binding missing for {item['path']}."
            for item in pack["file_coverage"] if item["status"] != "explicit"]


def require_authoring_evidence(state, plan, paths, *, evidence_ids=()):
    """The one mutation gate for legacy, staged, repair and primitive actions.

    Importing user-owned binary assets is a separate operator-authorized path.
    There is deliberately no policy switch which permits an LLM write without
    an observed syntax excerpt. Explicit references are not native validation.
    """
    if state is None:
        raise ValueError("Authoring requires durable observed syntax evidence, not a stateless write.")
    if plan is None:
        from types import SimpleNamespace
        plan = SimpleNamespace(required_case_files=list(paths),
            implementation_evidence_bindings=[SimpleNamespace(path=p, evidence_ids=list(evidence_ids)) for p in paths],
            openfoam_version=str(getattr(state, "openfoam_version", "unknown")))
    else:
        # Only the files mutated in this operation need to fit this check. The
        # complete design is checked independently before bundle commit.
        from types import SimpleNamespace
        bindings = {item.path: item for item in plan.implementation_evidence_bindings}
        plan = SimpleNamespace(required_case_files=list(paths), openfoam_version=plan.openfoam_version,
            implementation_evidence_bindings=[bindings[p] if p in bindings else
                SimpleNamespace(path=p, evidence_ids=list(evidence_ids)) for p in paths])
    pack = implementation_evidence_pack(state, plan, max_chars=None)
    missing = [item["path"] for item in pack["file_coverage"] if item["status"] != "explicit"]
    if missing:
        raise ValueError("Observed, explicit syntax evidence required before authoring: " + ", ".join(missing))
    return pack


def authoring_prompt_evidence(state,plan=None):
    """Same observed body/hash pack for primitive, legacy, staged and repair turns."""
    from types import SimpleNamespace
    paths = set(getattr(plan,"required_case_files",[]) or [])
    for record in state.engineering_evidence_records:
        for item in _walk(record.payload):
            if item.get("content") or item.get("content_excerpt"):
                paths.update(item.get("target_case_files",[]))
    projected=SimpleNamespace(required_case_files=sorted(paths),
        implementation_evidence_bindings=[item for item in getattr(plan,"implementation_evidence_bindings",[]) if item.path in paths],
        openfoam_version=getattr(plan,"openfoam_version","unknown"))
    pack=implementation_evidence_pack(state,projected,max_chars=None)
    # A primitive write may name an observed reference in evidence_ids without
    # any earlier target_case_files hint. Its body must still reach the stateless
    # authoring call. Unscoped reads are mandatory until explicitly file-scoped.
    observed,targets=observed_syntax(state,projected.openfoam_version)
    scoped={eid for ids in targets.values() for eid in ids}
    present={item["evidence_id"] for item in pack["records"]}
    pack["records"].extend(item for eid,item in observed.items() if eid not in present and eid not in scoped)
    return pack


def authoring_evidence_failures_for_representations(state, plan, *, raw_paths, typed_paths=(), block_mesh_path=None):
    """Return only authoring-evidence failures that cannot be replaced by deterministic serialization.

    Raw free-form OpenFOAM text remains evidence-gated. Typed dictionaries and the
    structured blockMesh DSL are generated by deterministic serializers and are therefore
    allowed to proceed to parser/native validation without a source excerpt for every file.
    This is the DEFERRED evidence path: syntax confidence is established by deterministic
    rendering plus downstream validation rather than by forcing design-time document reads.
    """
    raw = list(dict.fromkeys(raw_paths))
    if not raw:
        return []
    try:
        require_authoring_evidence(state, plan, raw)
    except ValueError as exc:
        return [str(exc)]
    return []
