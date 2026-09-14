# OpenFOAM Agent v4.9.3

## Region-aware native case build graph

v4.9.3 fixes the multi-region mesh routing defect exposed by the battery/heater solid heat-transfer case. The controller now binds authored `system/<region>/blockMeshDict` files to `blockMesh -region <region>` and compiles final `checkMesh -region <region>` validation for every Agent-declared named region. Root single-region behavior remains unchanged.

The change is generic: region names are read from Agent-owned plan metadata and authored case paths, while Python only validates the namespace and binds it to deterministic OpenFOAM native tool contracts.
