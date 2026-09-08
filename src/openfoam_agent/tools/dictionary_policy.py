"""Token-aware safety checks and explicit bounded include materialization."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path


_TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^\s{}();/]+|[{}();/]')


def tokens(text: str) -> list[str]:
    return [x.group(0) for x in _TOKEN.finditer(text) if not x.group(0).startswith(("//", "/*"))]


def library_entries(text: str) -> list[str]:
    values = tokens(text)
    libraries = []
    expect_key = True
    parentheses = 0
    index = 0
    while index < len(values):
        token = values[index]
        if token in {"{", "}", ";"} and parentheses == 0:
            expect_key = True
        elif token == "(":
            parentheses += 1
        elif token == ")":
            parentheses = max(0, parentheses - 1)
        elif expect_key and parentheses == 0:
            expect_key = False
            if token.strip("\"'") == "libs":
                cursor = index + 1
                if cursor >= len(values) or values[cursor] != "(":
                    raise ValueError("libs requires an explicit parenthesized library list.")
                cursor += 1
                while cursor < len(values) and values[cursor] != ")":
                    name = values[cursor].strip("\"'")
                    if not re.fullmatch(r"lib[A-Za-z][A-Za-z0-9_]*\.so", name):
                        raise ValueError("Dynamic, path-based, or malformed library names are forbidden.")
                    libraries.append(name)
                    cursor += 1
                if cursor + 1 >= len(values) or values[cursor:cursor + 2] != [")", ";"]:
                    raise ValueError("Malformed libs entry.")
                index = cursor + 1
                expect_key = True
        index += 1
    return libraries


_INCLUDE = re.compile(r'(?m)^\s*#include\s+"([^"\n]+)"\s*;?\s*$')


def materialize_local_includes(text: str, *, parent: Path, root: Path,
                               max_bytes: int = 1_000_000, max_depth: int = 12) -> tuple[str, list[dict[str, str]]]:
    """Flatten literal includes only after operator opt-in; native code sees no includes.

    Environment expansion, includeEtc, optional includes, codeStream, path traversal,
    symlinks, cycles, and oversized expansion are deliberately not interpreted.
    """
    root = root.resolve()
    records: list[dict[str, str]] = []
    consumed = 0

    def expand(body: str, directory: Path, stack: tuple[Path, ...]) -> str:
        nonlocal consumed
        if len(stack) > max_depth:
            raise ValueError("Include depth limit exceeded.")
        # Remove comments before matching directives, but never reinterpret quoted text.
        def replace(match):
            nonlocal consumed
            name = match.group(1)
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "$" in name or "\\" in name:
                raise ValueError("Unsafe include target.")
            raw = directory / relative
            path = raw.resolve()
            if not path.is_relative_to(root) or raw.is_symlink() or any(p.is_symlink() for p in raw.parents if p != root.parent):
                raise ValueError("Include escapes its explicitly trusted tree or uses a symlink.")
            if path in stack:
                raise ValueError("Include cycle detected.")
            size = path.stat().st_size
            consumed += size
            if consumed > max_bytes:
                raise ValueError("Expanded include byte budget exceeded.")
            data = path.read_bytes()
            records.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
            return expand(data.decode("utf-8"), path.parent, (*stack, path))
        result = _INCLUDE.sub(replace, body)
        if re.search(r"#\s*include", result, re.IGNORECASE):
            raise ValueError("Only standalone literal #include directives are supported for materialization.")
        if len(result.encode()) > max_bytes:
            raise ValueError("Expanded include byte budget exceeded.")
        return result
    return expand(text, parent.resolve(), ()), records
