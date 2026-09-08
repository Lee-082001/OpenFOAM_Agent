# v2.10.3 — Typed Dictionary Collision Recovery

- Fixed typed OpenFOAM dictionary container/scalar collisions such as redundant `boundaryField = {}` plus `boundaryField.inlet.type`.
- Container-only placeholders are normalized away because dotted leaf paths already define the block structure.
- Real scalar/block collisions now become deterministic engineering observations and trigger another LLM prepare turn instead of crashing the whole workflow.
- Compact engineering prompts now explicitly require leaf-only typed dictionary assignments; container blocks are implicit.
- Interactive CLI banner now reads `openfoam_agent.__version__` instead of the stale hard-coded `v2.8.0` string.
