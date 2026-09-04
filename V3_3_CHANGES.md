# OpenFOAM Agent v3.3.0 changes

## Native Foundation 13/14 installation discovery

- Production command authority is no longer the historical Python `SafeRunner` command list.
- A sourced `WM_PROJECT=OpenFOAM`, `WM_PROJECT_VERSION=13|14`, trusted `WM_PROJECT_DIR` is discovered into a sanitized `InstalledOpenFOAMIR`.
- Every executable found in trusted `FOAM_APPBIN` / OpenFOAM PATH entries is available as an Agent-selectable native application without adding a per-command Python wrapper.
- `$FOAM_MODULES` is scanned for installed solver modules; runtime-selectable model/function-object source evidence is discovered from the sourced tree.
- Documented v13/v14 capability metadata lives in the static capability graphs as fallback evidence. It is not inserted into `InstalledOpenFOAMIR` and never grants executable authority.
- Dynamic command authority is disabled for other project names/versions. The historical minimal command set remains only for offline/unit-test compatibility.

## Version-matched capability evidence

- The release ships both `config/openfoam13_capability_graph.json` and `config/openfoam14_capability_graph.json`.
- CLI default capability graph follows `WM_PROJECT_VERSION`; explicit `--capability-db` remains supported.
- `CapabilityCatalog` merges static documented providers with the current `InstalledOpenFOAMIR` and emits deterministic installed provider/evidence IDs for arbitrary discovered applications, solver modules and fvModels.
- One unreadable installed reference file/root is isolated rather than failing the entire evidence batch.

## General native utility pipeline

- Added `NativeOpenFOAMCommand` and `native_pipeline` to case-plan, repair, runtime-repair and strategy-revision contracts.
- Any application discovered in the trusted Foundation installation can participate in preprocessing/meshing/initialization; Python does not maintain a feature enum for those commands.
- Generic native execution is shell-free, workspace-confined, path-traversal checked, and prevents Agent-owned `-case`/root redirection.
- Existing specialized wrappers remain for deterministic output interpretation and compatibility, not as the universe of usable OpenFOAM tools.

## Execution IR and multi-region support

- Added `OpenFOAMExecutionSpec` and `RegionSolverAssignment`.
- `foamRun` is represented as a driver plus one selected solver module.
- `foamMultiRun` is represented as a driver plus explicit region-to-solver-module assignments.
- Direct/legacy solver applications may be selected as the execution driver when present in the trusted installation.
- Runtime orchestration now executes the selected driver rather than unconditionally invoking `foamRun`.
- Safety validation semantically parses `controlDict.regionSolvers` and requires exact agreement with multi-region execution IR.
- Runtime progress/log naming follows the selected driver.

## Security boundary

Dynamic discovery broadens engineering capability without broadening process trust. Before every launch Python re-resolves the bare executable under the trusted OpenFOAM root, uses `shell=False`, confines cwd to the case workspace, filters environment/path entries, rejects parent traversal and out-of-workspace absolute paths, and owns case/root routing. User/site app bins are not implicitly trusted.

## Regression coverage

New tests exercise both Foundation 13 and 14 fake installations, arbitrary newly discovered utilities and source modules, non-OpenFOAM/unsupported-version rejection, installed/static capability merging, generic command sandboxing, `foamRun`/`foamMultiRun` execution IR, semantic `regionSolvers` validation, and version-selected capability profiles.
