from __future__ import annotations

from .model import MeshIR, MeshPatch, PatternState
from .parser import find_boundary_list_start, parse_in_groups, parse_top_level_blocks


def parse_mesh_boundary(text: str) -> MeshIR:
    start = find_boundary_list_start(text)
    if start is None:
        return MeshIR()
    section = parse_top_level_blocks(text, start, ")")
    patches: list[MeshPatch] = []
    for entry in section.entries:
        if entry.key.pattern_state != PatternState.LITERAL:
            continue
        patches.append(
            MeshPatch(
                name=entry.key.value,
                patch_type=entry.declared_type,
                groups=parse_in_groups(entry.body),
                order=entry.order,
            )
        )
    return MeshIR(tuple(patches))
