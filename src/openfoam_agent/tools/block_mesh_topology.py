from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfoam_agent.schemas.engineering import TypedBlockMeshFile


# Hexahedron face/edge connectivity is expressed in terms of the eight block-local
# vertex slots used by blockMesh.  Validation below intentionally compares canonical
# vertex sets, not winding, because the first safety question is topological ownership:
# a declared boundary face must be a real exterior block face and must not already be
# owned by another patch.  Native blockMesh remains authoritative for geometric
# orientation/quality checks that require OpenFOAM's full implementation.
_HEX_FACE_SLOTS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)
_HEX_EDGE_SLOTS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _canonical_face(vertices: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(sorted(vertices))


def _canonical_edge(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True)
class BlockFaceOwner:
    block_index: int
    local_face_index: int
    oriented_vertices: tuple[int, int, int, int]


@dataclass(frozen=True)
class BlockMeshTopologyIssue:
    code: str
    message: str
    path: str = "system/blockMeshDict"


@dataclass(frozen=True)
class BlockMeshTopologyIR:
    """Deterministic topological interpretation of a TypedBlockMeshFile.

    ``face_owners`` maps canonical four-vertex faces to the block-local faces that
    own them.  One owner means exterior, two owners means conformal internal face,
    and more than two owners is non-manifold topology.
    """

    face_owners: dict[tuple[int, int, int, int], tuple[BlockFaceOwner, ...]]
    block_edges: frozenset[tuple[int, int]]

    @classmethod
    def from_spec(cls, spec: "TypedBlockMeshFile") -> "BlockMeshTopologyIR":
        face_owners_mut: dict[tuple[int, int, int, int], list[BlockFaceOwner]] = {}
        block_edges: set[tuple[int, int]] = set()
        for block_index, block in enumerate(spec.blocks):
            vertices = tuple(block.vertices)
            for local_face_index, slots in enumerate(_HEX_FACE_SLOTS):
                oriented = tuple(vertices[slot] for slot in slots)
                key = _canonical_face(oriented)
                face_owners_mut.setdefault(key, []).append(
                    BlockFaceOwner(
                        block_index=block_index,
                        local_face_index=local_face_index,
                        oriented_vertices=oriented,
                    )
                )
            for first_slot, second_slot in _HEX_EDGE_SLOTS:
                block_edges.add(_canonical_edge(vertices[first_slot], vertices[second_slot]))
        return cls(
            face_owners={key: tuple(value) for key, value in face_owners_mut.items()},
            block_edges=frozenset(block_edges),
        )


@dataclass(frozen=True)
class BlockMeshTopologyReport:
    ir: BlockMeshTopologyIR
    issues: tuple[BlockMeshTopologyIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def failures(self) -> list[str]:
        return [item.message for item in self.issues]

    def render(self) -> str:
        if self.valid:
            return "blockMesh topology contract passed."
        return "\n".join(f"- [{item.code}] {item.message}" for item in self.issues)


def validate_block_mesh_topology(spec: "TypedBlockMeshFile") -> BlockMeshTopologyReport:
    """Validate blockMesh topology before native execution.

    This check is deliberately generic and semantic rather than geometry-template
    specific.  It proves invariants that blockMesh itself requires regardless of
    whether the Agent chose an O-grid, Cartesian multi-block mesh, channel mesh, etc.:

    * every hex uses eight distinct vertex indices;
    * every declared patch face is a real block face;
    * a boundary face has exactly one block owner (never an internal shared face);
    * one exterior face is not assigned to multiple patch entries;
    * explicit curved/line edges correspond to actual block edges;
    * mergePatchPairs reference distinct, declared patch names.

    OpenFOAM remains the oracle for geometric orientation, curved-edge realization,
    grading compatibility and mesh quality.
    """

    ir = BlockMeshTopologyIR.from_spec(spec)
    issues: list[BlockMeshTopologyIssue] = []

    for block_index, block in enumerate(spec.blocks):
        if len(set(block.vertices)) != 8:
            issues.append(
                BlockMeshTopologyIssue(
                    code="degenerate_block_vertices",
                    message=(
                        f"block[{block_index}] must reference eight distinct vertices; "
                        f"observed {tuple(block.vertices)}."
                    ),
                )
            )

    for face, owners in sorted(ir.face_owners.items()):
        if len(owners) > 2:
            owner_ids = [item.block_index for item in owners]
            issues.append(
                BlockMeshTopologyIssue(
                    code="non_manifold_block_face",
                    message=(
                        f"face {face} is shared by {len(owners)} blocks {owner_ids}; "
                        "a conformal blockMesh face may have at most two owners."
                    ),
                )
            )

    assigned_faces: dict[tuple[int, int, int, int], tuple[str, int]] = {}
    for patch in spec.boundary:
        for face_index, face in enumerate(patch.faces):
            if len(set(face)) != 4:
                issues.append(
                    BlockMeshTopologyIssue(
                        code="degenerate_boundary_face",
                        message=(
                            f"boundary patch {patch.name!r} face[{face_index}] {tuple(face)} "
                            "must contain four distinct vertex indices."
                        ),
                    )
                )
                continue
            key = _canonical_face(face)
            owners = ir.face_owners.get(key, ())
            if not owners:
                issues.append(
                    BlockMeshTopologyIssue(
                        code="boundary_face_not_on_block",
                        message=(
                            f"boundary patch {patch.name!r} face[{face_index}] {tuple(face)} "
                            "does not correspond to any face of any declared hex block."
                        ),
                    )
                )
                continue
            if len(owners) != 1:
                owner_ids = [item.block_index for item in owners]
                issues.append(
                    BlockMeshTopologyIssue(
                        code="boundary_face_is_internal" if len(owners) == 2 else "boundary_face_non_manifold",
                        message=(
                            f"boundary patch {patch.name!r} face[{face_index}] {tuple(face)} "
                            f"has {len(owners)} block owners {owner_ids}; boundary faces must be "
                            "exterior faces with exactly one owner."
                        ),
                    )
                )
                continue
            previous = assigned_faces.get(key)
            if previous is not None:
                previous_patch, previous_index = previous
                issues.append(
                    BlockMeshTopologyIssue(
                        code="boundary_face_assigned_twice",
                        message=(
                            f"boundary patch {patch.name!r} face[{face_index}] {tuple(face)} "
                            f"duplicates exterior face already assigned to {previous_patch!r} "
                            f"face[{previous_index}]."
                        ),
                    )
                )
            else:
                assigned_faces[key] = (patch.name, face_index)

    defined_edges: dict[tuple[int, int], int] = {}
    for edge_index, edge in enumerate(spec.edges):
        if edge.start == edge.end:
            issues.append(
                BlockMeshTopologyIssue(
                    code="degenerate_block_edge",
                    message=f"edge[{edge_index}] starts and ends at vertex {edge.start}.",
                )
            )
            continue
        key = _canonical_edge(edge.start, edge.end)
        if key not in ir.block_edges:
            issues.append(
                BlockMeshTopologyIssue(
                    code="edge_not_on_block",
                    message=(
                        f"edge[{edge_index}] ({edge.start}, {edge.end}) is not an edge of any "
                        "declared hex block; blockMesh curved/explicit edges must refine a block edge."
                    ),
                )
            )
        if key in defined_edges:
            issues.append(
                BlockMeshTopologyIssue(
                    code="edge_defined_twice",
                    message=(
                        f"edge[{edge_index}] ({edge.start}, {edge.end}) duplicates edge "
                        f"definition edge[{defined_edges[key]}]."
                    ),
                )
            )
        else:
            defined_edges[key] = edge_index

    patch_names = {patch.name for patch in spec.boundary}
    for pair_index, (first, second) in enumerate(spec.merge_patch_pairs):
        if first == second:
            issues.append(
                BlockMeshTopologyIssue(
                    code="merge_patch_self_pair",
                    message=f"mergePatchPairs[{pair_index}] references {first!r} twice.",
                )
            )
        missing = [name for name in (first, second) if name not in patch_names]
        if missing:
            issues.append(
                BlockMeshTopologyIssue(
                    code="merge_patch_unknown_patch",
                    message=(
                        f"mergePatchPairs[{pair_index}] references undeclared patch name(s): "
                        + ", ".join(repr(name) for name in missing)
                    ),
                )
            )

    return BlockMeshTopologyReport(ir=ir, issues=tuple(issues))
