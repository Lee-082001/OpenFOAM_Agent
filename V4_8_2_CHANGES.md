# OpenFOAM Agent v4.8.2

v4.8.2 hardens the failure-local recovery architecture introduced in v4.8.0 and used by v4.8.1. It does not weaken native OpenFOAM validation and does not move CFD choices into Python.

## Bounded RepairEpisode memory

- The authoritative `RepairEpisode` remains controller state, but it is no longer serialized wholesale into every repair prompt.
- Model-visible repair memory is projected into a bounded root-failure digest/summary, current failure, implicated-file list, repair-history summary, and support-evidence window.
- Reference/support observations are archived separately from model context. The repair prompt sees at most a small top-k character-bounded window.
- Duplicate support observations are suppressed and reference-search events use smaller durable digests.
- The prompt builder applies support/history limits before focused-file partitioning, so reducing files is no longer the only context-pressure mechanism.

## Native diagnostic sufficiency gate

When an actual OpenFOAM diagnostic already reports an unknown/invalid value and explicitly enumerates supported alternatives, and implicated case files are available, the controller marks the next step `direct_repair_first`.

Python does **not** select an alternative. The Engineering Agent still decides the CFD/OpenFOAM value. The controller only prevents an unnecessary reference/capability retrieval from expanding context before one direct repair attempt. Retrieval reopens after a repair has been attempted for the current validation generation, or when a new native failure creates a new generation.

## Preserved validation behavior

`blockMesh`, `checkMesh`, zero-step consumer validation, CaseDeltaGraph validation, and subsequent native revalidation remain authoritative. v4.8.2 changes recovery memory and retrieval routing, not validation strictness.
