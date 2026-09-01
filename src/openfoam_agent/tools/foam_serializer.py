from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from openfoam_agent.schemas.engineering import FoamDictionaryEntry, TypedFoamDictionaryFile


class FoamSerializationError(ValueError):
    """Raised when a typed OpenFOAM dictionary specification is inconsistent."""


def serialize_foam_dictionary(spec: TypedFoamDictionaryFile) -> str:
    """Serialize path/value pairs into deterministic OpenFOAM dictionary syntax.

    Engineering values remain Agent-owned. Python only renders braces and semicolons,
    which removes a large class of syntax-only LLM output and repair turns.
    ``value`` is an OpenFOAM value expression (for example ``uniform (1 0 0)`` or
    ``Gauss linear``); dictionary nesting is represented by ``path`` components.
    """

    root: OrderedDict[str, object] = OrderedDict()
    for entry in spec.entries:
        _insert(root, entry)
    lines: list[str] = []
    _render_mapping(root, lines, indent=0)
    text = "\n".join(lines).rstrip() + "\n"
    return text


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
