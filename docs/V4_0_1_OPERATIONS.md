# OpenFOAM Agent 4.0.1: operational contracts

This candidate extends rc1 in six areas. The accompanying Korean implementation
report and `verification/v4_0_1/` describe the actual qualification level. Old
`docs/V4_AUDIT_33.json` and top-level `verification/*` are retained **rc1 history**,
not evidence that rc2 ran native CFD. There was no OpenFOAM/MPI/live-model run.

## Installation and regression reproduction

Use a separate source folder; retain the rc1 archive/workspaces. Python >=3.12 is
required. Only Python 3.13.5 was used for this release qualification.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q --junitxml=local-results.xml
python scripts/check_rc2_contract_examples.py
```

These install commands may access package indexes. The release's wheel smoke
used `--no-deps` and inherited the existing environment; it was not a clean
internet dependency-resolution test.

## Authoring and preserved requirements

CLI engineering defaults to staged design/authoring. When its authoring JSON
exceeds the configured **character** budget, the controller greedily assigns
required files to bounded tasks. It retains the full plan outside the model,
projects file-specific confirmed facts plus transitive provenance, repeats
unmapped/global constraints and interfaces, and supplies observed syntax reads.
A required file is assigned exactly once. Task responses cannot substitute a
new plan. No file/native work is committed until all tasks assemble; the existing
full-bundle semantic/syntax preflight still runs before commit. Every accepted
partial response is checkpointed.

The queue consumes the existing LLM/tool budgets. It does not silently raise
limits. An indivisible file/global contract still fails before a model call.
Automatic planning-stage partitioning and splitting a single generated file's
contents are not claimed. Library users must explicitly enable
`compact_phase_schemas=True, staged_case_authoring=True`; legacy mode retains
fail-closed behavior on oversized mandatory context.

Every generated write or patch, including legacy sequence, runtime repair and
revision paths, now passes `require_authoring_evidence`. A real reference read
can name `target_case_files`; a primitive write can alternatively supply its
observed `evidence_ids`. Search snippets, invented IDs and name matching alone
are insufficient. Full distinct observed read windows are retained. Unbound
reads are included in legacy authoring prompts so explicit IDs do not reference
invisible syntax. This certifies observed provenance, **not native compatibility**.
Operator-approved asset import is a separate path, not an LLM authoring bypass.

## Mesh dependency graph

`mesh-dependencies.json` records versioned native operation inputs, outputs,
regions, parents and command metadata. Ancestor closure includes derived assets
and literal case-local includes, then hashes current contents plus missing-file
sentinels. New/deleted files inside dependency roots change the digest. The
checkpoint binds the graph itself. All centralized case mutation and native
mesh writer paths invalidate stale regional evidence, seal and solve approval.

Only a narrowly recognized blockMesh invocation gets a restricted read set.
Unknown tools/options conservatively depend on all case input roots; a model's
claimed read set cannot narrow that policy. This trades extra revalidation for
safety. It is not OS syscall tracing or inference of arbitrary external data
reads. Existing include authoring restrictions still apply; dependency inspection
of a literal include is not permission to generate an otherwise disallowed file.

## Strict Linux isolation (explicit operator setup)

Default mode is `local_no_os_isolation`. It is not a filesystem/network sandbox.
The strict mode is selected with an operator-owned JSON policy:

```bash
openfoam-agent --interactive --backend codex --capability-db config/openfoam13_capability_graph.json --workspace /mnt/ofa-bounded-volume/runs --isolation-policy /path/to/isolation.policy.json
```

Adapt `examples/v4rc2/isolation.policy.example.json`. An administrator/operator
must already have provided an **unprivileged account**, system Bubblewrap,
a delegated cgroup v2 parent with cpu/memory/pids controllers, `cgroup.kill`, and
a separate writable filesystem whose **total capacity** does not exceed
`max_case_bytes`. An ordinary directory on a large shared filesystem is not a
hard quota. The agent does not mount filesystems, enable delegation, use sudo,
create school-server network listeners or fall back to unisolated execution.

Namespaces restrict mounts, PID/user/IPC/UTS/network/cgroup views; only public
runtime paths and explicitly authorized runtime roots are read-only exposed.
The case, temporary files and shared-memory files live on the bounded volume.
The wrapper joins its cgroup before exec. All ranks/descendants share aggregate
memory/swap/task/CPU-rate controls; CPU-time, cumulative wall and output budgets
are recorded/monitored. Whole-tree cancellation uses cgroup.kill. Workspace
locking prevents concurrent native runners from spending the same ledger.
Unknown historic aggregate CPU is not treated as zero.

Cumulative CPU/wall monitoring is polled, not a hard real-time deadline; small
overshoot is possible. `cpu.max` is a rate limit, not a total CPU-time counter.
Disk quota bounds current stored data/capacity, not lifetime I/O bandwidth. GPU,
scheduler/Slurm, arbitrary-device and I/O-rate quotas are not implemented. This
release did not execute the strict backend against a real writable cgroup or
Bubblewrap. Kernel integration/security qualification is still required.

Reports separate requested isolation from observed per-process isolation modes.
`--dry-run` does not turn an unavailable strict environment into a qualified one.

## Local MPI restart, preserving partial outputs

Use a checkpoint from the same workspace and compatible environment. Prepare
without model authentication or a solver run:

```bash
openfoam-agent --resume /path/to/run --prepare-parallel-restart latest --capability-db config/openfoam13_capability_graph.json --json
```

An explicit literal time may replace `latest`. The command selects a complete
common time across exactly the approved `processorN` ranks, validates static
mesh addressing, required field sizes/class/dimensions/patch names/finite values,
and binds the snapshot to the decomposition/module/argument/intake contract.
Incomplete newer outputs are **moved into restart-history, never force-deleted**.
An intent journal precedes changes; interrupted preparation is not auto-replayed.
Only top-level controlDict startFrom/startTime are rewritten; original requested
start/end conditions remain in the completion contract. Fresh approval is needed:

```bash
openfoam-agent --resume /path/to/run --solve --backend codex --capability-db config/openfoam13_capability_graph.json
```

Review `parallel-restart.json`, `restart-history/` and the prepared state before
that second command. Do not combine `--prepare-parallel-restart` with `--solve`.
A live/reused PID, nonempty/unknown cgroup or uncertain process identity blocks
preparation. It does not kill unrelated processes or guess that work completed.
Existing rank data are verified and reused; decomposePar is skipped. Final
reconstruction includes the positive saved times requested by native quantities
or conservation checks, not only the solver's final time. Sealed `0/` is never
rewritten by this reconstruction path; analyses needing time zero require its
original saved fields. Missing rank output blocks reconstruction.

Supported: local MPI, static meshes, uncollated processorN layout, bounded
ASCII/gzip volume/surface/point fields in the explicitly supported classes.
Unsupported or unqualified: moving topology, binary/collated/distributed fields,
Slurm, remote rank trees, custom hidden restart state or automatic rc1 checkpoint
migration. A `4.0.0rc1` source patch is not a checkpoint-migration promise.

## Native physical quantities and conservation

Attach `native_temperature.quantity.json` to the plan's `quantities_of_interest`.
Native scalar field observations read saved files and mesh geometry directly,
validate seven-base SI dimensions, exact region/patch selections, field classes,
element counts, finite values and reductions. Supported reductions are volume
mean/integral, area mean/integral, signed patch sum and cell extrema; time
statistics use bounded piecewise-linear interpolation without extrapolation.
Names and SI units must match the declared observable. Incompressible kinematic
pressure is not silently treated as Pa, and volume flux is not silently kg/s.

Attach `mass_conservation.check.json` to `conservation_checks`. `rhoPhi` and `rho`
are illustrative names: they must actually be saved with matching dimensions at
all requested times. A plan cannot turn an arbitrary CSV into physical proof.
The equation evaluated on each saved interval is:

`d(storage)/dt + sum(outward boundary flux) - integrated source = 0`.

Mass, energy and volume balances have distinct dimensions. All boundary patches
participate. Storage density and source density are explicitly declared, or the
approved plan explicitly assumes steady storage/zero source. Declared interfaces
are checked on both sides using outward flux signs at every saved sample. Every
interval and interface must pass; large opposing errors cannot disappear in a
single whole-window mean. Missing/stale/failed required quantity or balance
results make the report unsuccessful and prevent high confidence.

Supported geometry is static, reconstructed, ASCII/gzip scalar evidence with
bounded lists, planar convex faces and positively oriented, geometrically and
topologically closed cells. This subset intentionally refuses warped/unsupported
mesh formats instead of producing falsely certified volumes. The checker does
not independently establish the completeness of all PDE/model source terms,
pointwise interface continuity, turbulence/thermophysical model correctness or
unsaved timestep conservation. `physical_semantics_verified` has this narrow
field/dimension/selection/reduction scope; `governing_equation_completeness_verified`
remains false. Generic table arithmetic still reports semantics false.

## Primary implementation references (not runtime test evidence)

- Linux cgroup v2: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
- Bubblewrap option contract: https://raw.githubusercontent.com/containers/bubblewrap/main/bwrap.xml
- OpenFOAM 13 parallel workflow: https://doc.cfd.direct/openfoam/user-guide-v13/running-applications-parallel
- Control dictionary: https://doc.cfd.direct/openfoam/user-guide-v13/controldict
- Mesh description: https://doc.cfd.direct/openfoam/user-guide-v13/mesh-description
- File format/dimensions: https://doc.cfd.direct/openfoam/user-guide-v13/basic-file-format
- Mesh files: https://doc.cfd.direct/openfoam/user-guide-v13/mesh-files
- Reconstruction source: https://cpp.openfoam.org/v13/reconstructPar_8C_source.html
- Time selection: https://cpp.openfoam.org/v13/timeSelector_8H_source.html
