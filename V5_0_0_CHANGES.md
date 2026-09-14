# OpenFOAM Agent v5.0.0

## Authority & Evidence Hardening

v5.0.0 is an architectural hardening release. It does not add case-specific CFD templates. Its purpose is to make the ownership boundary explicit and testable:

> The Engineering Agent chooses CFD engineering. Python/controller code observes, validates, seals, authorizes, and executes; it does not invent CFD meaning.

### 1. Installed availability is no longer CFD semantic evidence

Installed solver modules, fvModels, and applications are discovered as identity/presence only. Name-based Python mappings such as `solid -> heat_transfer.conjugate` and `heatSource -> heat_generation.volumetric` are removed. Physics semantics come from the versioned documented capability graph or explicitly observed OpenFOAM references/source. Installation evidence may strengthen a documented provider's availability level, but cannot create physics semantics by itself.

### 2. User delegation and Agent autonomy are separated

`exploratory_completion_authorized` is no longer promoted to an unconditional `authorized=True` controller claim. User delegation applies to omitted concrete physical/model defaults that instantiate the requested problem. Solver, mesh, boundary-condition strategy, and numerical strategy remain Agent-owned engineering choices and do not require a fabricated user-consent flag.

### 3. One multi-region authority

For modern multi-region plans, `execution.regions` is the canonical named-region and solver-assignment topology. `region_layouts` and manifest-path inference remain compatibility fallbacks for older persisted plans only.

### 4. Shared region-aware native compiler semantics

Initial authoring and repair/revision now share deterministic region-scope helpers. Regional edits such as `system/battery/blockMeshDict` compile to `blockMesh -region battery`, and mesh-affecting regional repairs finish with `checkMesh -region <region>` for every canonical execution region. Region-aware `snappyHexMesh` preconditions inspect `constant/<region>/polyMesh/boundary`.

### 5. Agent-owned solve-input manifest is authoritative

Pre-solve validation no longer silently injects solver-specific `fvSchemes`, `fvSolution`, or initial-field requirements. `EngineeringPlan.required_case_files` declares the Agent-selected solve inputs. The controller verifies the declared files, boundary coverage, headers, and the explicitly declared root `system/controlDict` required for bounded execution.

### 6. Intent identity is separated from implementation evidence

Frozen intake digest plus `confirmed_fact_ids` provide immutable identity closure. Python no longer fabricates a `ConfirmedFactBinding(problem_interpretation)` for every fact. `ConfirmedFactBinding` is optional and means a truthful implementation assertion only. Binding paths cannot silently enlarge the Agent-owned required-file manifest.

### 7. Unknown native non-zero exits are not invented CFD failures

Explicit OpenFOAM fatal diagnostics remain case failures. Known infrastructure/tool failures remain inconclusive. An otherwise unclassified non-zero native exit is now `inconclusive/tool`, so Python cannot launch a CFD repair merely from an exit code.

### 8. Exact user-evidence locators

`source=user` facts receive controller-issued immutable evidence locators: source kind/index, source SHA-256, and exact character span. Review-critical inferred facts no longer claim dependency on every direct user fact; conservative dependency falls back to `request.summary`.

### 9. Evidence source identity and observation identity are distinct

Stable evidence IDs continue to identify canonical sources. `ObservedEngineeringEvidence.observation_sha256` binds the exact bounded descriptor issued in a run, preventing a source ID from being mistaken for immutable observed content.

### 10. Documentary syntax evidence is explicit-only

Filename/basename similarity is no longer promoted to implementation evidence. Documentary syntax evidence is advisory provenance; workspace safety, structured serialization/parsing, and native OpenFOAM consumers own execution authorization.

### 11. Native tool contracts are auditable data

Native command effect/prerequisite/phase contracts moved from a Python registry literal to `openfoam_agent/data/native_tool_contracts.json`. This data is execution metadata only; it does not choose a CFD strategy. Unknown commands gain no hidden authority.

### 12. Reference search is semantically neutral

The deterministic OpenFOAM reference index no longer expands Korean/English CFD terms into hidden aliases such as `열전달 -> heat transfer thermo`. Query translation and engineering search intent belong to the Agent.

## Migration safety

The supplied `apply_v500.py`:
- requires an exact v4.9.4 project version;
- checks exact source anchors before mutation;
- validates every transformed Python file with `ast.parse` and JSON payloads with `json.loads`;
- refuses unexpected collisions with new v5 payload files;
- supports a no-write `--preflight-only` mode;
- can emit a git-compatible patch against the exact inspected local tree with `--emit-patch`;
- creates a pre-migration tar backup before writes;
- does not modify `.git`;
- compiles the resulting package and can run targeted v5 regression tests.
