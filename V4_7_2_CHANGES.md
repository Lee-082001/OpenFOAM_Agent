# OpenFOAM Agent v4.7.2 — Native Toolchain Boundary

v4.7.2 fixes a live Foundation OpenFOAM 13 failure where `snappyHexMesh` exited with code 127 because `libscotch.so` was not visible to the sanitized runtime environment. The failure was incorrectly classified as a CFD case failure, which triggered an unnecessary LLM repair and a secondary context-budget error.

## Changes

- Separate trusted executable roots from trusted library roots.
  - Executables remain confined to the trusted OpenFOAM project tree.
  - Library-only trust may include a validated `WM_PROJECT_INST_DIR` / sibling `WM_THIRD_PARTY_DIR` tree.
  - `FOAM_EXT_LIBBIN`, `FOAM_LIBBIN`, and Scotch/Boost/CGAL/FFTW roots are retained only when they lie under a validated library trust anchor.
  - Arbitrary user-controlled library paths remain filtered from `LD_LIBRARY_PATH`.
- Add bounded native ELF dependency inspection under the exact sanitized environment used for OpenFOAM execution.
  - Explicit `=> not found` dependencies are reported before case mutation.
  - Non-ELF or unavailable dependency inspection is non-authoritative and does not invent a failure.
- Add controller-owned native toolchain preflight to initial authoring, prepare repair, runtime repair, and mesh-strategy revision before transactional mutation.
- Reclassify executable/loader failures as infrastructure rather than CFD case failures:
  - shared-library loader failures,
  - symbol lookup errors,
  - executable format errors,
  - exit codes 126/127 when the trusted executable could not start reliably.
- Infrastructure/toolchain failures never enter the LLM CFD repair loop. The native diagnostic remains available to the user/operator.
- Real `FOAM FATAL ERROR` / `FOAM FATAL IO ERROR` diagnostics remain case failures and stay repair-eligible.

## Live failure addressed

The motivating live run reached a valid controller-owned 13-file transient Y-branch case, passed `surfaceCheck` and `blockMesh`, then failed before `snappyHexMesh` application startup with:

```
error while loading shared libraries: libscotch.so: cannot open shared object file
```

That condition is now a native-toolchain/infrastructure boundary, not a CFD authoring defect.
