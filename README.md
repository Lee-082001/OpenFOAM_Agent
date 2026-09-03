# OpenFOAM Agent v3.0.1



> **v3.0.1 backend compatibility:** Claude and Codex structured-output schemas are now compiled per backend. Claude fixed tuples no longer emit unsupported `prefixItems`; Codex strict schemas explicitly require every property and remove transport-only defaults while retaining final Pydantic validation.

## v3.0.0: OpenFOAM semantic PreSolve layer

v3.0 replaces literal-name boundary validation with an explicit OpenFOAM semantic interpretation layer. `FoamDictionary` entries preserve key token form and ordering, `MeshIR` preserves patch type/group metadata, and `BoundaryFieldInterpreter` resolves effective patch fields using the OpenFOAM v13 order **exact patch -> patchGroup -> automatic empty -> regex/wildcard**. Overlapping groups/patterns retain dictionary ordering, quoted literals are not blindly classified as regex, and unsupported dynamic selectors become `INDETERMINATE` warnings rather than false missing-patch failures.

The same effective boundary resolution now drives both boundary coverage and constraint-patch validation, so PreSolve validates what OpenFOAM will effectively apply instead of comparing two sets of literal names. Resolution objects retain match kind, certainty and trace evidence for deterministic diagnosis. This is the first consumer of the new `verification/foam_semantics/` layer; future OpenFOAM semantics can extend that layer without adding ad-hoc parsers to `presolve.py`. See `V3_CHANGES.md`.


## v2.19.0: Claude Code subscription backend

A fourth autonomous model transport is available as `--backend claude`. It invokes an already installed Claude Code CLI in non-interactive print mode (`claude -p`) and requires `claude auth status` to report the Claude subscription/OAuth path (`authMethod=claude.ai`). Startup and model subprocesses remove API/provider routing variables such as `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, and Bedrock/Vertex/Foundry selector variables so an exported credential cannot silently turn this backend into a different billing/provider path. `CLAUDE_CODE_OAUTH_TOKEN` is intentionally preserved because Anthropic documents it as a subscription OAuth credential for headless use.

Claude Code is model transport only. Every call runs from an empty temporary working directory with `--no-session-persistence`, mandatory `--safe-mode`, `--tools ""`, and `--strict-mcp-config`; OpenFOAM Agent never asks Claude Code to read/edit the CFD case or execute native tools. Structured output is requested with `--output-format json --json-schema ...`, then the returned `structured_output` is independently revalidated by Pydantic with one bounded protocol-repair attempt. Role routing supports `CLAUDE_MODEL`, `CLAUDE_INTAKE_MODEL`, `CLAUDE_ENGINEERING_MODEL`, `CLAUDE_POSTPROCESS_MODEL`, and `CLAUDE_REVIEW_MODEL`; omitted model IDs delegate selection to the Claude Code CLI default. See `V2_19_CHANGES.md`.


## v2.18.0: runtime exit invariants, boundary consistency, and Codex backend

v2.18 closes the transient `RUNTIME_REPAIR` state as an internal orchestration invariant. Runtime repair now returns an explicit `RuntimeRepairDecision` (`RETRY_SOLVER`, `NEEDS_USER_REVIEW`, `BLOCKED`, or `STRATEGY_REVISION`) instead of overloading one boolean, and neither the Engineering Agent nor RuntimeOrchestrator may return `RUNTIME_REPAIR` to the top-level workflow. A defensive top-level handler converts any future leak into an explicit engineering block rather than the misleading `No v2 handler for RUNTIME_REPAIR` failure.

Pre-solve validation now checks cross-file constraint-patch compatibility. If the current `constant/polyMesh/boundary` marks a patch `empty`, `wedge`, `symmetry`, `symmetryPlane`, `cyclic`, or `cyclicAMI`, every required initial field must carry the same constraint patch type. Normal mesh `wall`/`patch` boundaries are intentionally not exact-matched against field BC classes such as `fixedValue` or `zeroGradient`. `blockMesh`, `snappyHexMesh`, and `createPatch` invalidate prior mesh/checkMesh/pre-solve evidence after execution attempts; runtime-repair topology or solver-input mutations also invalidate the current CaseSeal until `retry_solver` re-runs the required gates and reseals the exact case.

A third autonomous model transport is available as `--backend codex`. It invokes an already installed and ChatGPT-authenticated Codex CLI through `codex exec` using an ephemeral session, an isolated temporary working directory, a read-only sandbox, and `--output-schema`/`--output-last-message`. API-key routing environment variables are removed for this backend so it cannot silently become the ordinary API backend; the CLI's ChatGPT/Codex login remains available. Codex structured output is still revalidated by Pydantic, with one bounded protocol-repair attempt. CFD filesystem writes, OpenFOAM tools, safety gates, CaseSeal, and human approvals remain owned by deterministic OpenFOAM Agent Python. See `V2_18_CHANGES.md`.


## v2.17.0: mesh tool contracts and strategy escalation

v2.17 separates local dictionary repair from meshing-strategy revision. `system/blockMeshDict` now has a dedicated `block_mesh` typed DSL for vertices, blocks, edges and boundary patch lists, so generic dotted dictionary serialization no longer has to approximate blockMesh-specific list syntax. Deterministic Python also exposes narrow executable contracts such as `snappyHexMesh` requiring a fully 3D base mesh during snapping. If an observed `empty` patch proves that prerequisite is not met, the native command is not consumed and the next Engineering contract is a dedicated strategy revision. Repeated identical normalized native mesh failures also escalate instead of permitting endless local patch loops. Python never selects the replacement CFD/meshing strategy; the Engineering Agent does.

## v2.16.0: compact artifact-pointer semantic evidence

v2.16 keeps the v2.15 provenance and semantic-fidelity contract but stops asking the model to duplicate long case excerpts inside `EngineeringPlan`. Structural assertions now prefer `path + entry_path + expected_value`, while numeric relations point to actual case values with `entry_path` or a short raw-file anchor. Python extracts values from the current case before checking the Agent-submitted relation. Legacy v2.15 excerpt assertions remain readable. The CLI output ceiling is 24000 tokens as a safety margin for complete-case Structured Output; normal token use is still governed by compact schemas and model output.


OpenFOAM Agent v2 is an **agent-owned CFD engineering system with deterministic safety gates**.

The central design rule is:

> The agent chooses and designs. Python validates authority, safety, integrity, bounded execution, and evidence.

v2 is intentionally not a collection of hand-written CFD case templates. There is no production rule such as `if vortex shedding -> square obstacle template`, `if static -> snappyHexMesh`, or `if prescribed deformation -> displacementLaplacian`.




## v2.15.0: semantic fidelity contract (L2 foundation)

v2.15 adds the first production **L2 Semantic Fidelity** contract on top of the existing execution-integrity gates. New IntakeAgent-issued definitions use `semantic_contract_version=2`. Review-critical routing interpretations such as `classification.problem_type` and `temporal.behavior` may be marked `source=user` only when the user's exact evidence explicitly states that normalized interpretation; otherwise deterministic provenance handling demotes the attribution to `derived` without changing the Agent's chosen value. The CLI highlights these as `derived/review-critical` before `/confirm`, because confirmation makes the interpretation immutable downstream. In particular, a cylinder being geometrically inside a rectangular computational domain is not treated as direct user evidence of `internal_flow`.

Confirmed-fact bindings can now carry **case semantic assertions** (Agent-selected snippets that must actually be present in the current `0/`, `constant/`, or `system/` artifacts) and a CFD-agnostic **numeric relation assertion**. For a direct user numeric physics/scale/property target such as `Re=1000`, the Agent supplies the actual case scalar tokens and the generic numerator-product/denominator-product relation; individual terms may carry an Agent-selected numeric multiplier so an actual geometry token such as radius `0.5` can participate as `2×0.5`. Python only re-reads those artifacts and recomputes the submitted arithmetic against the confirmed target. Comment-only prose is stripped before evidence matching, so an Agent cannot satisfy the contract merely by writing `// Re=1000` or another self-claim into a case file. Python does not choose the formula, reference velocity, length scale, viscosity, solver, BCs, or mesh. New semantic-contract-v2 cases require case assertions for confirmed classification/temporal facts and a numeric relation for direct single-number physics/scale/property facts. These checks run again after repair, so a stale semantic assertion cannot silently authorize a modified case.

The new contract is backward compatible: legacy v1 intakes remain loadable, and empty v2.15 assertion fields are excluded from legacy canonical digests so existing confirmed-intake hashes and CaseSeals do not become stale merely because the software was upgraded. Assertion prose is whitespace-normalized during verification, and incomplete non-safety assertion payloads become normal deterministic failures rather than large Structured Output/Pydantic crashes. Regression suite at v2.15: **200 passed**. See `V2_15_CHANGES.md`.

## v2.14.0: evidence-gap lifecycle + protocol-resilience audit

v2.14 hardens the LLM/Python protocol after production runs showed that harmless Structured Output shape mistakes could still terminate an otherwise valid CFD workflow. Evidence gaps now have an explicit lifecycle. A gap ID is single-use: after deterministic retrieval it becomes `evidence_available` or `stagnant`; proceeding to case execution marks available gaps `satisfied`. If more external evidence is genuinely needed, the Agent must declare a new, more-specific gap with `refines_gap_id`, which supersedes the earlier gap. Repeating an already retrieved gap is refused without consuming another retrieval-cycle budget. Query strings, opaque gap IDs, whitespace, overlong query prose, invalid retrieval scope and read-count formatting are normalized deterministically because they are retrieval protocol metadata rather than CFD engineering decisions.

OpenAI Structured Output now has one bounded **protocol-repair attempt**, analogous to the existing local-model repair layer. If `responses.parse` raises a Pydantic validation error, the adapter asks the same model once to correct only the schema/protocol shape while explicitly preserving confirmed CFD facts. This protection applies to Intake, Engineering, Post-processing and Feedback/Review because they share the adapter; it does not bypass deterministic workspace safety, confirmed-fact coverage, solver/provider consistency, case sealing, pre-solve, native evidence or human approval gates. Repair schemas also accept syntactic no-op payloads so they can be rejected as normal deterministic unsuccessful actions rather than exploding into large union-validation traces. See `V2_14_CHANGES.md`.

## v2.13.0: evidence-gap retrieval + grouped runtime delta repair

v2.13 replaces free-form prepare search loops with **evidence-gap-driven batch retrieval**. The Engineering Agent must distinguish user-required unknowns, Agent-owned engineering choices, and genuinely external tool/version evidence gaps. Only the last category may invoke `gather_evidence`. One LLM turn can request up to four independent gaps; deterministic Python searches the capability graph and trusted installed OpenFOAM references in a batch, optionally reads a bounded number of top reference matches, and records canonical evidence IDs per gap. Repeating a gap is useful only when it adds new evidence: a zero-novelty retrieval marks that gap `stagnant`, and subsequent retrieval for the same gap is refused. A small retrieval-cycle hard fuse remains only as an emergency bound; once reached, the prepare Structured Output contract itself shrinks to `execute_case_plan | block`.

Runtime repair now has a separate `repair_runtime_case` contract. Exact edits are grouped per file (`file_patches[].edits[]`) and applied sequentially, so several legitimate fixes to one `fvSchemes`/`fvSolution` file no longer fail schema validation merely because the path appears more than once. Automatic runtime repair also receives a bounded native-error + relevant-case-file slice instead of the full engineering history/plan replay. The approved solver remains immutable and every resulting edit still passes the existing exact-match workspace safety, dictionary/pre-solve, mesh-freshness and resealing gates. Legacy `repair_case_plan` and retained-candidate patching also allow multiple sequential exact patches to the same file while still forbidding patch+replacement conflicts. See `V2_13_CHANGES.md`.


## v2.12.0: retained candidate plan + delta authoring repair

v2.12.0 keeps v2.11 transactional workspace authoring but no longer asks the LLM to regenerate the entire `execute_case_plan` after a pre-commit serialization or workspace-safety rejection. The complete rejected candidate remains only in Python memory; the retry contract accepts a small `repair_candidate_case_plan` delta (exact patch, replacement raw file, replacement typed dictionary, or dropped optional path) or `block`. Python applies that delta to the retained candidate, re-runs serialization and whole-bundle authoring preflight, and commits the full bundle only after it passes. This preserves all-or-nothing workspace semantics while avoiding large repeated Structured Output objects and the JSON truncation/token failure mode seen with complete-plan retries. Plan metadata repair remains a post-commit `RepairTurn` responsibility.

## v2.11.0: transactional case authoring and bounded complete-plan retry

v2.11.0 fixes the structural failure mode where an `execute_case_plan` could write the first few files, hit a deterministic workspace/serialization rejection on a later file, and then enter delta repair against a half-authored case. High-level case authoring is now **pre-commit transactional at the deterministic policy layer**: every raw and typed case file is rendered and checked for sandbox/content/library/size policy before the first candidate file is written. If authoring preflight fails, the workspace is left unchanged and the next compact LLM contract accepts only a corrected **complete** `execute_case_plan` or `block`; reference searches, file reads, and delta repair are intentionally unavailable because there is no partial case to repair. Complete-plan authoring failures have a separate 3-attempt bound, preventing them from consuming the full Engineering LLM budget. Once authoring preflight passes, the complete bundle is written before dictionary/native validation starts, so later OpenFOAM failures operate on a complete case and normal delta repair semantics remain valid. See `V2_11_CHANGES.md`.

## v2.10.5: plan-only repair and solver metadata consistency

v2.10.5 fixes a repair edge case discovered after a case had already passed dictionary validation, `blockMesh`, `checkMesh`, and pre-solve validation but the final `EngineeringPlan` contained inconsistent solver metadata. `repair_case_plan` now accepts an `updated_plan` without requiring a fake file mutation, so metadata-only repairs can be revalidated and sealed. The compact prepare prompt also requires `EngineeringPlan.solver` and `system/controlDict` to use the exact solver name of the selected observed capability provider and explicitly forbids placeholders such as `foamRunNameHere`, `solverName`, `TBD`, or `TODO`. Deterministic provider/case exact-match gates remain unchanged. See `V2_10_5_CHANGES.md`.

## v2.10.4: execution-plan robustness

v2.10.4 fixes an internal execution-plan expansion bug where a long parent `execute_case_plan.goal` could overflow the 200-character `rationale` limit on deterministic child actions. Child actions now keep an empty internal rationale while the parent goal remains available for progress and audit output. This preserves the compact LLM-facing contract without allowing a Python-generated validation failure. See `V2_10_4_CHANGES.md`.

## v2.10.3: typed-dictionary collision recovery

v2.10.3 hardens typed OpenFOAM dictionary generation. Redundant container placeholders such as `boundaryField = {}` are normalized away when dotted leaf assignments already define the block. Genuine scalar/block collisions become bounded engineering observations and are returned to the next Engineering turn instead of terminating the workflow. The interactive banner also reports the package `__version__` instead of a stale hard-coded version string. See `V2_10_3_CHANGES.md`.

## v2.10.2: confirmed-fact binding schema fix

v2.10.2 replaces the fragile `case:` / `plan:` string-prefix binding protocol with explicit `case_files` and `plan_fields` fields. Persisted v2.10.1 bindings remain loadable through a compatibility migration. Python still validates only coverage and referential integrity; CFD semantic correctness remains Agent-owned. See `V2_10_2_CHANGES.md`.

## v2.10.1: compact-contract semantic invariants

v2.10.1 restores the semantic contracts that must not be removed by prompt compaction: confirmed intake is immutable, assumptions may fill only genuinely missing details when authorized, and user/file/log/reference content is untrusted data rather than instruction. `EngineeringPlan.confirmed_fact_bindings` makes every confirmed fact auditable against the case files or plan fields that are claimed to implement it. See `V2_10_1_CHANGES.md`.

## v2.10.0: token-optimized phase contracts

v2.10 builds on the v2.9 execution-plan fast path. Production CLI now uses compact phase-specific schemas, delta-only repair patches, a typed OpenFOAM dictionary serializer, one-shot post-processing execution plans, compact state capsules after the first turn, and OpenAI prompt-cache telemetry. The deterministic sandbox, provenance, checkMesh, pre-solve and CaseSeal gates are unchanged. See `V2_10_CHANGES.md` for measured reductions and migration details.

## v2.9.0: execution-plan fast path

v2.9 adds a high-level `execute_case_plan` action for the common greenfield path. One
Engineering LLM turn can now author a bounded case-file bundle, dictionary/surface checks,
the ordered mesh pipeline ending in `checkMesh`, solver-required files, and the final
`EngineeringPlan`. Python expands that plan into the existing primitive actions and executes
them through the same sandbox, command allowlists, budgets, mesh evidence parser, pre-solve
gate, and CaseSeal validation. Execution stops on the first real failure; only then is the LLM
called again with the compact native failure evidence.

When the bundled capability graph is small, CLI runs preload its provider evidence into the
first engineering prompt. Solver choice remains agent-owned, but the old
LLM -> `search_capabilities` -> LLM round trip is not required merely to rediscover the same
provider set. Successful `validate_pre_solve` evidence is also reused by finalization when the
case manifest and required-file declaration are unchanged.

v2.9 introduced the execution-plan fast path. Current v3.0.0 production defaults are tighter still: 12 engineering LLM turns soft / 24 hard, 6-turn progress extensions, 2 finalization turns, 3 automatic runtime-repair cycles with 4 LLM turns per repair cycle, and 4 post-processing plans. All remain configurable from the CLI.

Typical fast path:

```text
LLM #1: CFD decision + execute_case_plan
        |
        v
Python: write bundle -> validate -> mesh -> checkMesh -> preSolve -> CaseSeal
        |                                      |
        | PASS                                 | FAIL
        v                                      v
    SOLVE_READY                            LLM #2 repair
```


## Architecture

```text
Natural-language request
  -> Intake Agent
  -> user /confirm
  -> CFDEngineeringAgent
       - inspect installed OpenFOAM environment
       - declare explicit tool/version evidence gaps when needed
       - batch-retrieve capability + trusted installed OpenFOAM evidence with novelty tracking
       - choose solver / mesh / BC / numerics / motion strategy
       - write and patch case files
       - run allowlisted OpenFOAM validation and mesh tools
       - read real failure logs and iterate
  -> deterministic safety + integrity gate
  -> MESH_READY
  -> deterministic pre-solve completeness gate
  -> SOLVE_READY
  -> user /solve
  -> foamRun
       - failure log -> same CFDEngineeringAgent -> bounded repair/retry
  -> runtime evidence
  -> CFDPostProcessingAgent
       - choose evidence-driven post-processing strategy
       - author isolated postprocessConfig dictionaries
       - run trusted foamPostProcess
       - inspect native time/postProcessing outputs
       - compute Cd/Cl/f/St from deterministic parsers when evidence exists
       - record advisory scientific-confidence/review focus
  -> RESULT_REVIEW_REQUIRED
       - /accept -> COMPLETE
       - /feedback <observation> -> Review Agent -> REVISION_READY
             - /confirm -> CFDEngineeringAgent revision -> MESH_READY -> /solve
```

### Agent-owned decisions

The engineering agent owns:

- steady/transient interpretation;
- solver selection;
- mesh strategy and geometry simplification;
- boundary conditions;
- properties and normalization;
- time step and numerical schemes;
- dynamic-mesh implementation;
- OpenFOAM error diagnosis and repair;
- post-processing strategy.

The semantic fields `temporal_behavior`, `motion_kind`, and `mesh_motion_requirement` remain as auditable common language. They do **not** trigger a Python implementation recipe.

### Deterministic responsibilities

Python owns only bounded execution and evidence boundaries:

- allowlisted OpenFOAM commands;
- workspace/path sandboxing;
- executable OpenFOAM directive and arbitrary code blocking;
- confirmed-user-fact provenance;
- solver identity consistency between `EngineeringPlan` and `system/controlDict`;
- native `foamDictionary`, `surfaceCheck`, and `checkMesh` evidence;
- SHA-256 sealing of every pre-solve execution input under `0/`, `constant/`, and `system/`, including native-generated mesh files;
- agent-step, solver-attempt, command-timeout, file-size, and mesh-cell limits;
- explicit `/confirm`, `/solve`, human `/feedback`, and `/accept` authority/review gates.

## What was removed in v2

The old production architecture was deleted rather than retained as a second hidden planning path. Removed components include:

- rule-based case templates and `case_factory`;
- deterministic requirement -> physics -> equation -> solver routing chain;
- `CapabilityGapPlanner` as a solver decision-maker;
- deterministic runtime `deltaT` repair planner;
- solver source-generation factory and verified-registry path from the v0.x prototype;
- motion-specific Foundation case contracts such as fixed `displacementLaplacian` or BC requirements;
- Phase-specific evaluation code and tests that asserted those old decisions.

The capability graph remains, but only as **read-only evidence available to the agent**.

## Installation

Python 3.12+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

For native execution, source an installed OpenFOAM Foundation environment first. The bundled capability graph is Foundation v14; a different version can use `--capability-db` with a version-matched graph. The engineering agent can also search the installed `FOAM_TUTORIALS`, `FOAM_SRC`, and `FOAM_ETC` trees directly.

## CLI

### 1. Intake only, offline

The rule-based backend exists only as an offline intake regression baseline. It cannot generate a case.

```bash
openfoam-agent \
  "사각형 장애물 주위 vortex shedding Re=1000 나머지는 탐색용으로 정해줘"
```

This stops at `INTAKE_REVIEW_REQUIRED`.

### 2. Autonomous engineering with OpenAI

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."

openfoam-agent --interactive \
  --backend openai \
  --confirm-api-calls
```

`OPENAI_MODEL` / `--model` is only the global default. Intake, Engineering, Post-processing, and Review may use different model IDs; see **Role-based model routing** below.

### Local Ollama backend over SSH tunnel (v2.7.0)

The Agent can use an Ollama model through the same `StructuredLLM` boundary without changing intake, engineering, validation, OpenFOAM execution, repair, post-processing, or review orchestration. Ollama is only the text/reasoning backend.

For the school GPU server, keep Ollama bound to server loopback and create the SSH tunnel from the local PC/WSL:

```bash
ssh -fN -L 11434:127.0.0.1:11434 oslee@mlfm4.knu.ac.kr
```

Then run the Agent locally:

```bash
openfoam-agent --interactive \
  --backend ollama \
  --model gemma4:31b
```

Ollama defaults are:

```text
base_url = http://localhost:11434/v1
model    = gemma4:31b
api_key  = ollama  # OpenAI-compatible dummy value
```

`--base-url` and `OLLAMA_BASE_URL` can override the endpoint, but the Agent accepts only loopback hosts (`localhost`, `127.0.0.1`, `::1`). Direct URLs such as `http://mlfm4.knu.ac.kr:11434/v1` or `0.0.0.0` are rejected so the remote Ollama service stays behind SSH local forwarding.

At startup the Ollama backend calls `/v1/models` through the tunnel, verifies connectivity and the requested role models, and fails without any OpenAI fallback if the tunnel/service/model is unavailable. A typical connection error is:

```text
Cannot connect to Ollama at http://localhost:11434. Check that the SSH tunnel to mlfm4.knu.ac.kr and the Ollama service are running.
```

Configuration is available through:

```text
--model / OLLAMA_MODEL
--base-url / OLLAMA_BASE_URL
OLLAMA_API_KEY (optional; defaults to ollama)
--intake-model / OLLAMA_INTAKE_MODEL
--engineering-model / OLLAMA_ENGINEERING_MODEL
--postprocess-model / OLLAMA_POSTPROCESS_MODEL
--review-model / OLLAMA_REVIEW_MODEL
```

The Ollama adapter reuses the installed OpenAI Python SDK against the OpenAI-compatible `/v1/chat/completions` endpoint, but it deliberately does **not** send the full Pydantic Agent schema as an Ollama constrained-decoding grammar. Complex schemas such as `EngineeringTurn` can fail in local grammar initialization before the model runs. Ollama therefore uses generic JSON mode (`response_format={"type":"json_object"}`), includes the Pydantic JSON schema in the prompt as generation guidance, then performs authoritative `model_validate_json()` validation in Python. Invalid local-model output receives at most two structured-repair turns (three total generation attempts). Only a Pydantic-valid object proceeds to the existing Agent/safety/evidence gates. OpenAI keeps its existing strict Structured Outputs path unchanged.

For multi-turn Intake, v2.7.3 also gives only the Ollama/local adapter two bounded **semantic provenance** repair turns. If a local model marks a synthesized summary as `source=user` with translated/paraphrased evidence, the deterministic validator rejects it and the repair prompt re-supplies the exact user turns, the invalid draft, and the provenance rule. Multi-turn summaries should be corrected to `source=derived` with `reason`/`depends_on`; direct facts still require short verbatim evidence. Persistent fabricated evidence remains a terminal failure.

`--confirm-api-calls` is intentionally not required for `--backend ollama` because requests stay on local loopback and traverse the user-created SSH tunnel; that flag remains mandatory for the cloud OpenAI, Codex, and Claude backends.


### Codex CLI backend with ChatGPT login (v2.18.0)

`--backend codex` uses the locally installed Codex CLI as a model-only transport while retaining the same OpenFOAM Agent state machine and deterministic execution boundary. Authenticate the CLI separately with `codex login`; startup verifies `codex login status` and the required non-interactive `codex exec` flags before any Agent call. Explicit cloud authorization is still required:

```bash
openfoam-agent --interactive \
  --backend codex \
  --confirm-api-calls \
  --capability-db config/openfoam14_capability_graph.json
```

Role-specific model flags and `CODEX_MODEL`, `CODEX_INTAKE_MODEL`, `CODEX_ENGINEERING_MODEL`, `CODEX_POSTPROCESS_MODEL`, and `CODEX_REVIEW_MODEL` are supported. If no Codex model is specified, the adapter leaves model selection to the Codex CLI default. Each call is stateless/ephemeral, runs from an empty temporary directory under `--sandbox read-only`, and receives only the bounded prompt/state capsule supplied by OpenFOAM Agent. `OPENAI_API_KEY`, `CODEX_API_KEY`, API base URL, organization, and project routing variables are removed from the Codex subprocess environment by design. The final JSON is validated again by the same Pydantic contract before any deterministic action is permitted.

### Claude Code backend with Claude subscription login (v2.19.0)

Authenticate Claude Code separately and verify the subscription login before starting the Agent:

```bash
claude auth login
claude auth status

openfoam-agent --interactive \
  --backend claude \
  --confirm-api-calls
```

Startup requires `claude auth status` to report `loggedIn=true` and `authMethod=claude.ai`. The adapter strips `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, and Bedrock/Vertex/Foundry routing selectors before both the auth check and every model call. It then invokes `claude -p` with `--output-format json`, `--json-schema`, `--no-session-persistence`, mandatory `--safe-mode`, `--tools ""`, and `--strict-mcp-config` from an empty temporary directory. The model therefore receives only the bounded prompt/state capsule; case writes and OpenFOAM commands remain deterministic Python actions. Claude Code does not provide the same OS-level read-only sandbox flag used by Codex, so managed enterprise hooks remain an external installation-policy trust boundary rather than something OpenFOAM Agent claims to sandbox.

Role-specific environment variables are `CLAUDE_MODEL`, `CLAUDE_INTAKE_MODEL`, `CLAUDE_ENGINEERING_MODEL`, `CLAUDE_POSTPROCESS_MODEL`, and `CLAUDE_REVIEW_MODEL`. If no Claude model is specified, the adapter delegates model choice to the Claude Code CLI default.

### Role-based model routing

The workflow has four configurable LLM roles. `--backend` selects the backend for the **whole workflow**, while each role may select a different model ID within that backend. The current CLI does not mix OpenAI, Ollama, Codex, and Claude backends in one run.

| Role | CLI flag | Environment variable | Used for |
| --- | --- | --- | --- |
| Intake | `--intake-model` | `OPENAI_INTAKE_MODEL` / `OLLAMA_INTAKE_MODEL` / `CODEX_INTAKE_MODEL` | Natural-language intake, clarification, and solver-independent CFD problem definition |
| Engineering | `--engineering-model` | `OPENAI_ENGINEERING_MODEL` / `OLLAMA_ENGINEERING_MODEL` / `CODEX_ENGINEERING_MODEL` | Initial CFD engineering, case/mesh repair, **runtime repair**, and confirmed revision engineering |
| Post-processing | `--postprocess-model` | `OPENAI_POSTPROCESS_MODEL` / `OLLAMA_POSTPROCESS_MODEL` / `CODEX_POSTPROCESS_MODEL` | Post-processing planning/reporting after a successful solve unless `--skip-postprocess` is used |
| Review | `--review-model` | `OPENAI_REVIEW_MODEL` / `OLLAMA_REVIEW_MODEL` / `CODEX_REVIEW_MODEL` | Human-feedback diagnosis and revision proposal generation after `/feedback` |

Runtime repair and confirmed case revision intentionally reuse the **Engineering** model because they continue the same solver/mesh/BC/numerics responsibility. The Review model only diagnoses feedback and proposes changes; it does not execute the revision.

`--model` is the backward-compatible default model for every role. Resolution order for each role is:

```text
role-specific CLI flag
  > role-specific environment variable
  > --model
  > OPENAI_MODEL / OLLAMA_MODEL / CODEX_MODEL / CLAUDE_MODEL
  > backend built-in default (Ollama: gemma4:31b; Codex/Claude: CLI-selected model)
```

OpenAI has no built-in default model, so every OpenAI role must resolve either through a global default or an explicit role override. Codex and Claude may omit a model ID and delegate selection to their authenticated CLI defaults. A global default is optional when all four roles are configured explicitly. If multiple roles resolve to the same model ID, the CLI reuses one backend adapter instance for those roles.

#### OpenAI examples

**One model for every Agent:**

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="<model-id>"

openfoam-agent --interactive \
  --backend openai \
  --confirm-api-calls
```

Equivalent CLI form:

```bash
openfoam-agent --interactive \
  --backend openai \
  --confirm-api-calls \
  --model "<model-id>"
```

**Cost-aware routing: a lighter model for routine roles and a stronger model for Engineering:**

```bash
openfoam-agent --interactive \
  --backend openai \
  --confirm-api-calls \
  --model "gpt-5.6-luna" \
  --engineering-model "gpt-5.6-sol"
```

That resolves to:

```text
Intake           gpt-5.6-luna
Engineering      gpt-5.6-sol
Runtime repair   gpt-5.6-sol   # follows Engineering
Revision         gpt-5.6-sol   # follows Engineering
Post-processing  gpt-5.6-luna
Review            gpt-5.6-luna
```

**Explicit model for every role:**

```bash
openfoam-agent --interactive \
  --backend openai \
  --confirm-api-calls \
  --intake-model "gpt-5.6-luna" \
  --engineering-model "gpt-5.6-sol" \
  --postprocess-model "gpt-5.6-luna" \
  --review-model "gpt-5.6-luna"
```

The OpenAI backend currently uses one `OPENAI_API_KEY`/OpenAI endpoint configuration for the run; the per-role settings choose **model IDs**, not separate API keys or separate providers. Per-role OpenAI/Ollama/Codex/Claude backend mixing or per-role API keys/base URLs are not exposed by the current CLI.

#### Ollama role routing

The same four role flags work with `--backend ollama`:

```bash
openfoam-agent --interactive \
  --backend ollama \
  --intake-model "<local-intake-model>" \
  --engineering-model "<local-engineering-model>" \
  --postprocess-model "<local-postprocess-model>" \
  --review-model "<local-review-model>"
```

All Ollama roles use the same loopback `--base-url` / `OLLAMA_BASE_URL` and API-key setting. Startup health checking verifies every distinct requested role model before the workflow starts.

#### Verify the resolved routing

At startup the CLI prints the resolved models, for example:

```text
OpenAI model routing: intake=gpt-5.6-luna, engineering=gpt-5.6-sol, postprocess=gpt-5.6-luna, review=gpt-5.6-luna; runtime-repair/revision use the engineering model.
```

`LLM-CONTEXT` and `LLM-USAGE` progress events also include the model used for each call, and JSON reports expose the resolved `model_routes`.

Typical interactive flow:

```text
OpenFOAM Agent> 사각형 장애물 주위 vortex shedding Re=1000 나머지는 탐색용으로 정해줘
... intake review ...
OpenFOAM Agent> /confirm
... agent designs/writes/meshes/repairs and completes pre-solve validation until SOLVE_READY ...
OpenFOAM Agent> /solve
... bounded foamRun; failures are returned to the same engineering agent ...
... on success, bounded post-processing runs automatically ...
... RESULT_REVIEW_REQUIRED with artifacts, metrics and agent review focus ...
OpenFOAM Agent> /feedback 후류가 너무 대칭이고 vortex shedding이 이상해 보여
... diagnosis hypotheses + revision proposal; sealed case remains unchanged ...
OpenFOAM Agent> /confirm
... confirmed revision is applied, revalidated and resealed ...
# or use /reject while REVISION_READY to keep the current sealed case unchanged
OpenFOAM Agent> /solve
... revised run/post-processing ...
OpenFOAM Agent> /accept
... COMPLETE ...
```

Useful commands:

```text
/show
/details
/confirm
/solve
/feedback <mesh/result observation>
/accept
/reject
/set <fact>=<value>
/edit <text>
/undo
/run
/mode easy|guided|strict
/new
/exit
```

The first `/confirm` approves the immutable intake and bounded case/mesh preparation. It does **not** authorize the solver. `/solve` is a separate authority gate and is accepted only for a sealed `SOLVE_READY` case. After results are produced, `/feedback` records a human observation without mutating the case; the Review Agent produces an auditable revision proposal. A second `/confirm` is required before that proposal can modify the sealed case, and the revised case requires a fresh `/solve`. `/reject` discards the active proposal and returns to the unchanged mesh/result review state. Rejected proposals remain in audit history. `/accept` is the only transition from `RESULT_REVIEW_REQUIRED` to `COMPLETE`.

### Live progress after `/confirm` and `/solve` (v2.3)

v2.3 adds an observational `ProgressEvent` bus shared by engineering, runtime repair, post-processing, and human-feedback review. The default CLI mode is `--progress normal`, so a long `/confirm` no longer appears frozen while the Agent is working. Only observable workflow/tool facts are rendered; model rationale or private chain-of-thought is never printed.

```text
OpenFOAM Agent> /confirm
[ENGINEERING] 확정된 CFD 정의로 autonomous engineering 시작
[ENGINEERING 01/12] capability graph 조회: incompressibleFluid
[ENGINEERING 01/12] OK Capability search returned 1 provider(s).
[ENGINEERING 07/12] case 파일 작성: system/controlDict
[ENGINEERING 07/12] OK Wrote system/controlDict.
[ENGINEERING 08/12] mesh command 실행: checkMesh
[ENGINEERING 08/12] OK checkMesh returned status 0; evidence passed.
  cells=23760, maxNonOrtho=0, maxSkew=2.84e-13
[ENGINEERING 09/12] engineering plan 최종 검증 및 case seal
[ENGINEERING 09/12] OK Engineering plan accepted and case sealed.
```

If deterministic final validation rejects the plan, normal progress now exposes the bounded, path-redacted gate reason immediately instead of only showing a generic failure:

```text
[FINALIZING 01/2] FAIL Engineering plan rejected by deterministic safety/evidence gate.
  reason:
    - A successful current checkMesh result with cell-count evidence is required before solve approval.
```

These are deterministic validation messages, not model rationale or private reasoning.

`/solve` uses the same event bus. In `normal` mode, raw solver output is retained in log files while the terminal receives bounded live summaries of `Time`, Courant number, attempts, repair transitions, and post-processing actions. `verbose` additionally shows every Agent read/list action and raw `foamRun` stdout. Progress is written to stderr so `--json` stdout remains valid JSON.

```bash
# recommended default
openfoam-agent --interactive --backend openai --confirm-api-calls --progress normal

# no live progress
openfoam-agent --interactive --backend openai --confirm-api-calls --progress quiet

# every Agent action + raw foamRun output
openfoam-agent --interactive --backend openai --confirm-api-calls --progress verbose
```

### Token-aware model context (v2.4)

v2.4 keeps complete engineering/runtime/post-processing provenance locally but sends a compact working projection to the cloud model. Long residual histories, old tool outputs, result inventories, and feedback history are no longer retransmitted in full on every Agent turn.

Default model-facing hard caps are:

- engineering prompt: 60,000 characters;
- post-processing prompt: 40,000 characters;
- feedback-review prompt: 40,000 characters;
- engineering recent observations: 12 bounded excerpts;
- post-processing recent observations: 8 bounded excerpts;
- post-processing result inventory: 80 representative file entries;
- OpenAI response cap: `--llm-max-output-tokens 24000`.

Runtime residual evidence is projected as total sample count, latest residuals by recent field, and a short recent tail instead of every parsed residual sample. The original `RuntimeReport` remains unchanged locally. Deterministic plan/evidence validation also continues to use the full local event history rather than trusting the compact model projection.

Before each structured model call the progress stream reports approximate request size:

```text
[LLM-CONTEXT 07/12] prepare LLM context 준비
  promptChars=18342, schemaChars=9470, approxTokens=15120, compacted=False
```

`approxTokens` is deliberately conservative tokenizer-free telemetry; it is not OpenAI billing/usage data. When the OpenAI response includes usage metadata, v2.4 follows it with an exact provider-reported event such as `inputTokens=...`, `outputTokens=...`, and `totalTokens=...`. Use `--progress quiet` to hide these events. The explicit output-token cap is independently useful because a structured action normally needs far less than the model's maximum output window.

```bash
# default
openfoam-agent --interactive --backend openai --confirm-api-calls \
  --llm-max-output-tokens 24000

# increase only if a legitimately large single case-file action needs it
openfoam-agent --interactive --backend openai --confirm-api-calls \
  --llm-max-output-tokens 24000
```

### 3. One-shot preview without native OpenFOAM commands

```bash
openfoam-agent \
  "사각형 장애물 주위 vortex shedding Re=1000 나머지는 탐색용으로 정해줘" \
  --backend openai \
  --confirm-api-calls \
  --confirm-intake \
  --dry-run
```

The agent may author files, but `foamDictionary`, mesh utilities, and `foamRun` are not executed. The terminal state is `CASE_PREVIEW_READY`, not `MESH_READY`.

### 4. One-shot native preparation and solve

```bash
openfoam-agent \
  "사각형 장애물 주위 vortex shedding Re=1000 나머지는 탐색용으로 정해줘" \
  --backend openai \
  --confirm-api-calls \
  --confirm-intake \
  --solve
```

`--solve` requires `--confirm-intake` and cannot be combined with `--dry-run`.

## Automatic post-processing and human review (v2.2)

After a successful `foamRun`, v2.2 automatically hands the immutable solved case to `CFDPostProcessingAgent` unless `--skip-postprocess` is set. The post-processing agent decides what analysis is useful for the problem and can search installed OpenFOAM references, author isolated dictionaries under `postprocessConfig/`, run `foamPostProcess`, inspect native output files, and finish with an evidence-bound report.

For wake/vortex-shedding cases, the agent may choose vorticity and force-coefficient analysis. When a native `coefficient.dat` and the executed force-coefficient dictionary are available, deterministic Python computes mean Cd, mean/RMS Cl, a Cl-based shedding frequency, and Strouhal number. The LLM cannot invent these numerical metrics; each analysis is bound to SHA-256 hashes of both the coefficient file and the reference-scale dictionary. If there are too few cycles or the signal is not stable enough, the report keeps explicit limitations instead of claiming scientific validation.

Typical successful terminal output includes:

```text
runtime: success=True, attempts=1, lastTime=20.0, maxCo=...
postprocess: success=True, actions=..., native=...
forces: samples=..., meanCd=..., rmsCl=..., f=..., St=...
artifacts:
- 20/vorticity
- postProcessing/.../coefficient.dat
visualize: cd <case_dir> && paraFoam
```

Post-processing failure does **not** rewrite a successful solver run as a solver failure. Runtime success remains preserved and the post-processing report records its own limitations. The post-processing Agent may also provide an explicitly advisory `scientific_confidence`, reasons, and recommended human checks; those fields are never treated as deterministic proof. Whether post-processing succeeds or is partial, a successful solve ends at `RESULT_REVIEW_REQUIRED`, not `COMPLETE`. Use `--skip-postprocess` when only bounded execution is wanted; human review is still required.

### Human feedback and revision provenance

`/feedback <text>` is accepted at `MESH_READY` or `RESULT_REVIEW_REQUIRED`. The text is stored separately from the confirmed intake as `HumanFeedback` with its run/scope/evidence snapshot hash. The Review Agent can propose likely causes and changes, but cannot edit files or run OpenFOAM. The proposal is bound to the exact current EngineeringPlan digest and case-manifest digest.

If the feedback changes a confirmed user fact (for example `Re=1000 -> Re=500`), the Review Agent must route back to intake review instead of silently changing the case. Otherwise the proposal stops at `REVISION_READY`; only `/confirm` authorizes a new engineering round. A required case revision cannot finalize with an unchanged solver-input manifest. The accepted revision records added/removed/modified file hashes in `RevisionRecord`.

Before the revised run, the exact pre-revision `0/constant/system` inputs are copied to `revision-history/rev-XXXX/baseline_inputs/`, while old numeric time directories, `postProcessing/`, post-process configuration and tool logs are moved into the same private revision archive. This prevents stale outputs from being mixed with the new run while preserving both the original inputs and prior results for audit/rollback/comparison. Previous engineering events remain as provenance, but each human-confirmed revision receives a fresh bounded engineering/native/mesh-repair budget.

## Agent tools

The engineering agent receives bounded actions for:

- environment inspection;
- capability search;
- installed OpenFOAM reference search/read;
- case file list/read/write/delete;
- `foamDictionary` validation;
- `surfaceCheck`;
- `blockMesh`, `surfaceFeatureExtract`, `snappyHexMesh`, `createPatch`, `checkMesh`;
- final preview sealing;
- same-solver runtime retry or review-required blocking.

After runtime success, the post-processing agent separately receives bounded actions for:

- installed OpenFOAM post-processing reference search/read;
- write-only analysis configuration under `postprocessConfig/`;
- trusted `foamPostProcess`;
- read-only listing/reading of native numeric time directories and `postProcessing/`;
- deterministic force-coefficient analysis and evidence-bound finalization.

The process runner does not expose a shell. The executable allowlist is:

```text
blockMesh
surfaceFeatureExtract
surfaceCheck
snappyHexMesh
createPatch
checkMesh
foamRun
foamDictionary
foamPostProcess
```

## Security and confidentiality

The OpenAI backend is a cloud backend: after `--confirm-api-calls`, the confirmed CFD request and bounded engineering observations/log excerpts are intentionally transmitted to OpenAI. The API key is not inserted into prompts, local absolute paths are redacted from model-bound observations, and `store=False` is used by default. Native OpenFOAM subprocesses receive a sanitized environment and must resolve to executables inside the trusted OpenFOAM installation. Run directories/logs are private (`0700`/`0600`).

Operational success is evidence-bound: capability claims must come from observed searches, `MESH_READY` requires current trusted `checkMesh` evidence, and runtime success requires zero return status, `End`, actual `Time = ...` progress, and no fatal/non-finite evidence. Post-processing numerical metrics are derived from native outputs and hash-bound analysis dictionaries rather than accepted from model prose. Human revision proposals are additionally bound to the exact plan/case seal they reviewed and cannot mutate the case before `/confirm`. This prevents the model from merely *claiming* that tools, analyses, or revisions succeeded. It does not prove scientific correctness, mesh/time-step convergence, or experimental agreement; `COMPLETE` means the human explicitly accepted the reviewed result.

### Canonical engineering evidence IDs

v2.4.1 no longer lets `EngineeringPlan` invent evidence labels such as `tool_result:checkMesh at preparation step 98` or `user_fact:confirmed_intake`. Successful capability/reference tools create deterministic `ev_cap_<hash>` / `ev_ref_<hash>` records that are supplied under `available_evidence`; the model may only select those exact IDs. Confirmed intake, `checkMesh`, and the current case manifest are shown under `deterministic_bindings` and are validated automatically by Python rather than restated by the model.

See `SECURITY.md` for the threat model and residual limitations.

## Failure semantics

A failed mesh or dictionary tool is an **observation**, not an automatic terminal failure. The result is returned to the engineering agent, which may inspect files/references, modify the case, and retry inside bounded resource budgets. v2.11.0 measures the progress-aware engineering budget in **LLM turns**: production CLI defaults to a 12-turn soft boundary, 6-turn progress extensions and a 24-turn hard cap. Deterministic tool actions retain a separate cap, and all limits remain configurable. Repeating the same action/result loop does not earn more budget.

### Engineering action sequences (v2.8.0)

The Engineering Agent is no longer forced to spend one model call before every predictable tool action. A single `EngineeringTurn` may still contain one legacy action, or it may contain a bounded `sequence` of 2-6 ordered deterministic actions. Typical sequences are `write -> foamDictionary`, `write STL -> surfaceCheck`, `write mesh input -> blockMesh/snappyHexMesh -> checkMesh`, and solver-input construction ending in `validate_pre_solve`. Runtime repair can end a validated sequence with `retry_solver`.

Python does **not** execute a sequence as an unchecked batch. Every member passes through the existing workspace sandbox, command allowlist, native-command budget, mesh-freshness rules, pre-solve completeness checks, and safety/evidence gates. The first failed/rejected member stops the remaining sequence immediately and returns the real diagnostic to the next Engineering Agent turn. Rewriting the same file twice inside one sequence without an intervening validation/native action is rejected before execution.

Raw `EngineeringEvent` records remain local and complete for audit/provenance. Model context is different: consecutive events from one sequence are collapsed into one compact `engineering_sequence_summary`, with the failed diagnostic retained when applicable. Reports expose `engineering_llm_turns_used`, `engineering_tool_actions_used`, `engineering_sequences_used`, and `tool_actions_per_llm_turn` so token-efficiency improvements can be measured directly.

Preparation also has independent default limits of 40 executed native OpenFOAM validation/mesh commands and 6 mesh-repair cycles. If the engineering window ends immediately after a current passing `checkMesh`, a finalization-only window of 2 LLM turns lets the Agent submit finalization/blocking actions; it cannot use that window as an unbounded second engineering phase.

During runtime, a failed `foamRun` log is returned to the same Engineering Agent/model. The default policy allows up to 3 autonomous repair/retry cycles (4 solver executions total), with up to 4 Engineering LLM turns and 48 deterministic actions inside each repair cycle. Automatic repair cannot switch the already approved solver. Only mesh-affecting edits invalidate current `checkMesh` evidence. Changes to `fvSchemes`, `fvSolution`, `controlDict`, initial fields, and other non-mesh solver inputs keep the existing mesh evidence current and are validated by their appropriate dictionary/pre-solve checks. Changes to mesh-generation dictionaries, geometry inputs, or generated `constant/polyMesh` require a fresh `checkMesh` before `retry_solver` is accepted.


### Budget controls

The CLI exposes the production defaults so operators can tighten them for expensive environments:

```bash
--engineering-steps 12
--engineering-hard-cap 24
--engineering-extension 6
--finalization-steps 2
--engineering-tool-budget 160
--native-command-budget 40
--mesh-repair-cycles 6
--runtime-repair-cycles 3
--runtime-repair-steps 4
--runtime-repair-tool-budget 48
--postprocess-steps 4
--postprocess-native-budget 8
--skip-postprocess
```

These are resource/safety limits only. They do not encode solver, mesh, boundary-condition, or numerical-method choices.

## Evidence and integrity

At `MESH_READY`, v2 seals:

- the `EngineeringPlan` digest;
- every execution input file under `0/`, `constant/`, and `system/`;
- each file's SHA-256, size, and origin (`agent` or native OpenFOAM tool);
- the aggregate manifest SHA-256;
- current parsed `checkMesh` evidence.

`foamRun` is blocked if the sealed inputs or plan change after approval.

## Tests

The v3.0.0 release tree currently passes **239 regression tests**. The tests focus on architecture boundaries rather than preserving v0.x planner behavior. They cover:

- no rule-based engineering fallback;
- capability retrieval without deterministic solver planning;
- path/code/library sandboxing;
- strict OpenAI structured-output compatibility;
- trusted executable provenance and sanitized native subprocess environment;
- cloud-bound local-path redaction and trusted OpenFOAM reference roots;
- private workspace/log permissions;
- observed capability/evidence provenance and runtime progress evidence;
- no hidden required case-template files;
- solver/case mismatch and confirmed-fact provenance;
- mesh failure -> agent repair -> retry;
- mesh-scoped stale `checkMesh` invalidation without redundant reruns after solver-only edits;
- full pre-solve input sealing, including native mesh outputs;
- cell/resource bounds;
- runtime failure log -> same-agent repair;
- automatic runtime-success -> post-processing state transition;
- isolated postprocess configuration that cannot mutate the solve-input seal;
- hash-bound postprocess configuration execution and native result reads;
- deterministic Cd/Cl/shedding-frequency/Strouhal extraction from force-coefficient evidence;
- post-processing failure isolation that preserves successful runtime evidence;
- solver-change and `/solve` approval gates.
- adversarial dynamic-mesh end-to-end recovery, including unsafe-directive rejection, native mesh sealing, state rehydration, SIGFPE solver-input repair without redundant `checkMesh`, resealing, and same-solver retry.

See `ARCHITECTURE.md` and `V2_CHANGES.md` for the detailed boundary and migration notes.

### Minimal native preflight

When native execution is enabled, `/confirm` resolves only the trusted `checkMesh` executable before the engineering loop starts. If it is unavailable, the run stops as `ENGINEERING_BLOCKED` before any engineering LLM action is consumed. Other OpenFOAM utilities are discovered/used later according to the agent-selected strategy. `--dry-run` skips this check.

### Unified native failure diagnostics (v2.5.0)

Native failures no longer collapse to only `returned status 1`/`-6`. The same bounded `NativeFailureDiagnostic` path is used by `foamDictionary`, `surfaceCheck`, `blockMesh`, `surfaceFeatureExtract`, `snappyHexMesh`, `createPatch`, `checkMesh`, `foamRun`, and `foamPostProcess`. Complete stdout/stderr remains in the private workspace log; a bounded raw OpenFOAM diagnostic is surfaced to both the next relevant Agent turn and normal CLI progress. For example:

```text
[ENGINEERING 10/12] FAIL snappyHexMesh returned status 1; native diagnostic captured.
  reason:
    - nativeCommand: snappyHexMesh
    - returnCode: 1
    - diagnosticKind: foam_fatal_io_error
    - --> FOAM FATAL IO ERROR:
    - keyword locationInMesh is undefined ...
```

`foamRun` failures use the same projection before runtime repair and `foamPostProcess` failures use it before the next post-processing action. `foamDictionary` and `surfaceCheck` now also preserve full raw logs instead of existing only as transient tool output. Model/user-visible diagnostics are path-redacted and bounded. This is observation only: Python does not choose the engineering repair from the diagnostic.


### Pre-solve completeness validation (v2.5.1)

The interactive CLI no longer treats a passing `checkMesh` as sufficient proof that a case is ready for `foamRun`. The Engineering Agent declares the solver inputs it requires in `EngineeringPlan.required_case_files`; deterministic validation then checks those files, core system dictionaries, and boundary coverage before exposing `/solve`. A native run proceeds only after `SOLVE_READY`.

This keeps engineering choices with the Agent: Python does not infer that a particular solver needs `U`, `p`, or another field. It verifies the Agent's declared contract against the actual sealed case.
