# OpenFOAM Agent v3.0.2

## Canonical OpenFOAM case-file contract

v3.0.2 fixes a structural authoring defect in v3.0.1 where generic `TypedFoamDictionaryFile` serialization emitted only dictionary bodies while `blockMeshDict` had its own `FoamFile` header. The result could pass `foamDictionary -keywords` yet fail later when `blockMesh` or `foamRun` opened the file through OpenFOAM IOobject/regIOobject header handling.

### Deterministic header ownership

- Added `tools/foam_file.py` as the shared OpenFOAM file-contract layer.
- `serialize_foam_dictionary()` now always emits one canonical `FoamFile` header.
- `serialize_block_mesh()` uses the same header renderer instead of a separate hard-coded header.
- Header `object` is derived from the filename and `location` from the case-relative parent path.
- Typed text output owns `version 2.0` and `format ascii`.
- Ordinary typed system/constant files default to `class dictionary`.
- Initial fields under `0/` infer `volScalarField`, `volVectorField`, `volSphericalTensorField`, `volSymmTensorField`, or `volTensorField` from unambiguous `internalField` syntax.
- `TypedFoamDictionaryFile.foam_class` is available when a field/object class cannot be proven from the body.
- Legacy `FoamFile.object/class/location/format` typed leaves are consumed only as compatibility metadata. Matching values are de-duplicated; conflicts with the path-derived contract are rejected before writing.

### Transactional and PreSolve validation

- Complete `execute_case_plan` bundles validate solve-critical `FoamFile` contracts before any candidate file is committed.
- `validate_dictionary` now verifies the IOobject-facing header before invoking `foamDictionary` so a dictionary-syntax pass cannot mask a missing header.
- PreSolve validates the same header contract for core system files and Agent-declared required solve inputs.
- Header validation checks the top-level `FoamFile` block, `format`, `class`, `object`, object/path consistency, optional location consistency, system `class dictionary`, field-class validity, and unambiguous `internalField` class consistency.

### Systematic runtime diagnosis

Runtime-repair context now includes a deterministic batch scan of core system files, all current `0/` fields, and Agent-declared required solve inputs. If a legacy/raw case contains the same header defect across multiple files, the Engineering Agent sees the whole defect set in one repair cycle rather than burning one `foamRun` attempt per file.

### Regression coverage

Added regression tests for canonical system/field headers, scalar/vector class inference, explicit class fallback, legacy header de-duplication, conflicting object rejection, PreSolve header blocking before native dictionary validation, and batch runtime contract diagnosis.

Full test suite: **252 passed**.
