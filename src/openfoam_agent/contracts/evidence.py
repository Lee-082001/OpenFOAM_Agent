"""Implementation evidence is observed provenance, never inferred CFD authority."""
from __future__ import annotations

import hashlib

from openfoam_agent.llm.context import ContextBudgetError


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _observation_id(reference: str, record_id: str, excerpt_sha256: str) -> str:
    """Bind one immutable observation window to its source and durable record."""
    payload = f"{reference}\0{record_id}\0{excerpt_sha256}".encode("utf-8")
    return "obs_ref_" + hashlib.sha256(payload).hexdigest()[:24]


def observed_syntax(state, version="unknown"):
    """Project exact observed read windows without inventing file coverage.

    ``evidence_id`` remains the compatibility/source identity for existing state.  Each
    observed window additionally receives a content-bound ``observation_id`` so audits
    can distinguish two different excerpts from the same OpenFOAM source reference.
    """
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
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            observation_id = _observation_id(reference, record.record_id, digest)
            entry = available.setdefault(
                eid,
                {
                    "evidence_id": eid,
                    "source": reference,
                    "record_id": record.record_id,
                    "is_excerpt": True,
                    "openfoam_version": version,
                    "observed_windows": [],
                },
            )
            if not any(window["observation_id"] == observation_id for window in entry["observed_windows"]):
                entry["observed_windows"].append(
                    {
                        "record_id": record.record_id,
                        "content": body,
                        "excerpt_sha256": digest,
                        "observation_id": observation_id,
                    }
                )

    separator = "\n// --- separate observed read window ---\n"
    for entry in available.values():
        # Do not imply disjoint windows are a single continuous source excerpt.
        entry["content"] = separator.join(window["content"] for window in entry["observed_windows"])
        entry["excerpt_sha256"] = hashlib.sha256(entry["content"].encode("utf-8")).hexdigest()
        entry["observation_ids"] = [
            window["observation_id"] for window in entry["observed_windows"]
        ]
        offset = 0
        for window in entry["observed_windows"]:
            body = window.pop("content")
            window.update(start_char=offset, end_char=offset + len(body))
            offset += len(body) + len(separator)
    return available, targets


def implementation_evidence_pack(state, plan, *, max_chars: int | None = 16000):
    """Return only explicitly scoped implementation provenance.

    v5.0 removes basename/content guessing. A reference mentioning ``fvSolution`` or
    ``T`` is not evidence for a case file unless retrieval/controller state explicitly
    scoped that observation to the exact target path or the sealed plan names the same
    observed evidence ID. Missing documentary evidence is advisory and is never filled
    with a heuristic match.
    """
    available, targets = observed_syntax(state, plan.openfoam_version)
    bindings = {
        path: list(dict.fromkeys(ids))
        for path, ids in targets.items()
        if path in plan.required_case_files
    }
    bindings.update(
        {
            item.path: list(dict.fromkeys(item.evidence_ids))
            for item in plan.implementation_evidence_bindings
            if item.evidence_ids
        }
    )

    selected = {}
    coverage = []
    for path in plan.required_case_files:
        ids = list(bindings.get(path, []))
        missing = [eid for eid in ids if eid not in available]
        if missing:
            raise ValueError(
                f"Implementation evidence for {path} was not actually observed: {missing}"
            )
        for eid in ids:
            selected[eid] = available[eid]
        coverage.append(
            {
                "path": path,
                "evidence_ids": ids,
                "status": "explicit" if ids else "missing",
            }
        )

    for path in bindings:
        if path not in plan.required_case_files:
            raise ValueError(
                f"Implementation evidence binding references undeclared file: {path}"
            )

    total = sum(len(item["content"]) for item in selected.values())
    if max_chars is not None and total > max_chars:
        raise ContextBudgetError(
            "Explicit implementation evidence exceeds the authoring context budget; "
            "split the observed evidence projection. No evidence was truncated or inferred."
        )
    return {
        "records": list(selected.values()),
        "file_coverage": coverage,
        "complete": all(item["status"] == "explicit" for item in coverage),
        "scope": (
            "Explicit observed excerpts only. Missing documentary coverage is advisory; "
            "native/deterministic validation establishes executable compatibility."
        ),
    }


def evidence_coverage_failures(plan, state):
    """Audit helper; not an authoring/execution permission gate."""
    pack = implementation_evidence_pack(state, plan, max_chars=None)
    return [
        f"Explicit syntax provenance missing for {item['path']}."
        for item in pack["file_coverage"]
        if item["status"] != "explicit"
    ]


def require_authoring_evidence(state, plan, paths, *, evidence_ids=()):
    """Compatibility audit for callers that explicitly request documentary coverage.

    This function no longer represents the production mutation gate. Production
    authoring is authorized by workspace/security policy and validated by deterministic
    serializers/parsers/native consumers. Calling this helper intentionally requests a
    stricter documentary-provenance audit for the named paths.
    """
    if state is None:
        raise ValueError("Documentary evidence audit requires durable observed state.")
    from types import SimpleNamespace

    if plan is None:
        projected = SimpleNamespace(
            required_case_files=list(paths),
            implementation_evidence_bindings=[
                SimpleNamespace(path=path, evidence_ids=list(evidence_ids)) for path in paths
            ],
            openfoam_version=str(getattr(state, "openfoam_version", "unknown")),
        )
    else:
        existing = {item.path: item for item in plan.implementation_evidence_bindings}
        projected = SimpleNamespace(
            required_case_files=list(paths),
            openfoam_version=plan.openfoam_version,
            implementation_evidence_bindings=[
                existing[path]
                if path in existing
                else SimpleNamespace(path=path, evidence_ids=list(evidence_ids))
                for path in paths
            ],
        )
    pack = implementation_evidence_pack(state, projected, max_chars=None)
    missing = [
        item["path"] for item in pack["file_coverage"] if item["status"] != "explicit"
    ]
    if missing:
        raise ValueError(
            "Explicit observed syntax provenance required for this audit: "
            + ", ".join(missing)
        )
    return pack


def authoring_prompt_evidence(state, plan=None):
    """Project observed reference bodies for stateless authoring context."""
    from types import SimpleNamespace

    paths = set(getattr(plan, "required_case_files", []) or [])
    for record in state.engineering_evidence_records:
        for item in _walk(record.payload):
            if item.get("content") or item.get("content_excerpt"):
                paths.update(item.get("target_case_files", []))
    projected = SimpleNamespace(
        required_case_files=sorted(paths),
        implementation_evidence_bindings=[
            item
            for item in getattr(plan, "implementation_evidence_bindings", [])
            if item.path in paths
        ],
        openfoam_version=getattr(plan, "openfoam_version", "unknown"),
    )
    pack = implementation_evidence_pack(state, projected, max_chars=None)

    # Unscoped observations may still be useful context, but they are not counted as
    # file coverage and therefore cannot silently become implementation evidence.
    observed, targets = observed_syntax(state, projected.openfoam_version)
    scoped = {eid for ids in targets.values() for eid in ids}
    present = {item["evidence_id"] for item in pack["records"]}
    pack["records"].extend(
        item
        for eid, item in observed.items()
        if eid not in present and eid not in scoped
    )
    return pack


def authoring_evidence_failures_for_representations(
    state, plan, *, raw_paths, typed_paths=(), block_mesh_path=None
):
    """Compatibility hook for the former syntax-evidence mutation gate.

    Documentary evidence is advisory. Deterministic workspace policy,
    serializers/parsers and native validation own authoring authorization.
    """
    return []


def advisory_authoring_evidence_summary(state, plan=None, *, max_records: int = 8):
    """Compact, non-authorizing provenance summary for repair/revision prompts."""
    try:
        pack = authoring_prompt_evidence(state, plan)
    except (ValueError, ContextBudgetError):
        return {
            "records": [],
            "file_coverage": [],
            "complete": False,
            "scope": "Advisory only; deterministic/native validation owns authorization.",
        }
    records = []
    for item in pack.get("records", [])[:max_records]:
        records.append(
            {
                "evidence_id": item.get("evidence_id"),
                "observation_ids": item.get("observation_ids", []),
                "source": item.get("source"),
                "record_id": item.get("record_id"),
                "openfoam_version": item.get("openfoam_version"),
            }
        )
    return {
        "records": records,
        "file_coverage": pack.get("file_coverage", []),
        "complete": bool(pack.get("complete")),
        "scope": "Advisory provenance only; absence never authorizes or blocks a mutation.",
    }
