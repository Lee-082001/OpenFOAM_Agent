from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from openfoam_agent.schemas.engineering import (
    FoamDictionaryEntry,
    TypedBlockMeshFile,
    TypedFoamDictionaryFile,
)
from openfoam_agent.tools.block_mesh_topology import validate_block_mesh_topology
from openfoam_agent.tools.foam_file import (
    FoamFileContract,
    FoamFileContractError,
    render_foam_file_header,
    resolve_foam_file_contract,
)


class FoamSerializationError(ValueError):
    """Raised when a typed OpenFOAM dictionary specification is inconsistent."""


def serialize_foam_dictionary(spec: TypedFoamDictionaryFile) -> str:
    """Serialize path/value pairs into deterministic OpenFOAM dictionary syntax.

    Engineering values remain Agent-owned. Python only renders braces and semicolons,
    which removes a large class of syntax-only LLM output and repair turns.
    ``value`` is an OpenFOAM value expression (for example ``uniform (1 0 0)`` or
    ``Gauss linear``); dictionary nesting is represented by ``path`` components.
    """

    try:
        contract, body_entries = resolve_foam_file_contract(
            spec.path,
            entries=spec.entries,
            explicit_class=spec.foam_class,
        )
    except FoamFileContractError as exc:
        raise FoamSerializationError(str(exc)) from exc

    root: OrderedDict[str, object] = OrderedDict()
    entries = _normalize_entries(body_entries)
    for entry in entries:
        _insert(root, entry)
    lines: list[str] = []
    _render_mapping(root, lines, indent=0)
    body = "\n".join(lines).rstrip()
    return render_foam_file_header(contract) + "\n" + body + "\n"


_STRUCTURAL_CONTAINER_MARKERS = frozenset({"{}", "{ }", "{\n}", "block", "dictionary"})


def _normalize_entries(entries: Iterable[FoamDictionaryEntry]) -> list[FoamDictionaryEntry]:
    """Normalize harmless container placeholders before deterministic rendering.

    Typed dictionaries represent blocks implicitly through dotted leaf paths. Models may
    occasionally emit a redundant container placeholder such as ``boundaryField = {}``
    alongside ``boundaryField.inlet.type``. That carries no engineering value, so it is
    safe to discard. A real scalar/container collision is still rejected with an
    actionable diagnostic rather than silently changing CFD content.
    """

    items = list(entries)
    descendant_parents: set[str] = set()
    first_child: dict[str, str] = {}
    for entry in items:
        parts = entry.path.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            descendant_parents.add(parent)
            first_child.setdefault(parent, entry.path)

    normalized: list[FoamDictionaryEntry] = []
    for entry in items:
        if entry.path not in descendant_parents:
            normalized.append(entry)
            continue
        marker = " ".join(entry.value.strip().lower().split())
        if marker in _STRUCTURAL_CONTAINER_MARKERS:
            # Container structure is already implied by descendant leaf paths.
            continue
        child = first_child.get(entry.path, f"{entry.path}.<leaf>")
        raise FoamSerializationError(
            f"Typed dictionary path {entry.path!r} is used both as a scalar and as a block "
            f"(for example {child!r}). Container paths are implicit: omit the parent entry "
            f"and provide leaf paths such as {child!r}."
        )
    return normalized


def _insert(root: OrderedDict[str, object], entry: FoamDictionaryEntry) -> None:
    current: OrderedDict[str, object] = root
    components = entry.path.split(".")
    for component in components[:-1]:
        existing = current.get(component)
        if existing is None:
            child: OrderedDict[str, object] = OrderedDict()
            current[component] = child
            current = child
            continue
        if not isinstance(existing, OrderedDict):
            raise FoamSerializationError(
                f"Typed dictionary path collides with scalar entry at {component!r}."
            )
        current = existing
    leaf = components[-1]
    if leaf in current:
        raise FoamSerializationError(
            f"Typed dictionary contains duplicate/colliding path: {entry.path}"
        )
    current[leaf] = entry.value


def _render_mapping(mapping: OrderedDict[str, object], lines: list[str], *, indent: int) -> None:
    pad = " " * indent
    for key, value in mapping.items():
        if isinstance(value, OrderedDict):
            lines.append(f"{pad}{key}")
            lines.append(f"{pad}{{")
            _render_mapping(value, lines, indent=indent + 4)
            lines.append(f"{pad}}}")
        else:
            rendered = str(value).strip()
            if not rendered:
                raise FoamSerializationError(f"Typed dictionary value for {key!r} is empty.")
            # A value is an expression only; deterministic serializer owns the ';'.
            if rendered.endswith(";"):
                rendered = rendered[:-1].rstrip()
            lines.append(f"{pad}{key} {rendered};")


def serialize_block_mesh(spec: TypedBlockMeshFile) -> str:
    """Render the blockMesh-specific list/dictionary DSL deterministically.

    Geometry/topology choices remain Agent-owned, but Python proves generic topology
    invariants before serialization. This prevents known-invalid boundary ownership
    (for example, declaring an internal shared block face as a cylinder boundary) from
    reaching native blockMesh and turning into an expensive LLM repair loop.
    """

    topology = validate_block_mesh_topology(spec)
    if not topology.valid:
        raise FoamSerializationError(
            "blockMesh topology contract failed before native execution:\n" + topology.render()
        )

    def num(value: float) -> str:
        return format(float(value), ".16g")

    header = render_foam_file_header(
        FoamFileContract(
            path=spec.path,
            class_name="dictionary",
            object_name="blockMeshDict",
            location="system",
        )
    ).rstrip("\n")
    lines = [
        header,
        "",
        f"{spec.scale_keyword} {num(spec.scale)};",
        "",
        "vertices",
        "(",
    ]
    for vertex in spec.vertices:
        x, y, z = vertex.coordinates
        lines.append(f"    ({num(x)} {num(y)} {num(z)})")
    lines.extend([
        ");",
        "",
        "blocks",
        "(",
    ])
    for block in spec.blocks:
        verts = " ".join(str(index) for index in block.vertices)
        cells = " ".join(str(count) for count in block.cells)
        grading = block.grading.strip().rstrip(";")
        lines.append(f"    hex ({verts}) ({cells}) {grading}")
    lines.extend([
        ");",
        "",
        "edges",
        "(",
    ])
    for edge in spec.edges:
        definition = edge.definition.strip().rstrip(";")
        suffix = f" {definition}" if definition else ""
        lines.append(f"    {edge.kind} {edge.start} {edge.end}{suffix}")
    lines.extend([
        ");",
        "",
        "boundary",
        "(",
    ])
    for patch in spec.boundary:
        lines.extend([
            f"    {patch.name}",
            "    {",
            f"        type {patch.type.strip().rstrip(';')};",
            "        faces",
            "        (",
        ])
        for face in patch.faces:
            lines.append("            (" + " ".join(str(index) for index in face) + ")")
        lines.extend([
            "        );",
            "    }",
        ])
    lines.extend([
        ");",
        "",
        "mergePatchPairs",
        "(",
    ])
    for first, second in spec.merge_patch_pairs:
        lines.append(f"    ({first} {second})")
    lines.extend([
        ");",
        "",
    ])
    return "\n".join(lines)
