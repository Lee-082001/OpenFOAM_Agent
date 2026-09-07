from __future__ import annotations

import os
import re
import hashlib
from collections import OrderedDict
import unicodedata
from pathlib import Path


class OpenFOAMReferenceIndex:
    """Bounded read-only search over trusted installed official OpenFOAM trees.

    Environment-derived roots are accepted only when they resolve inside the
    current WM_PROJECT_DIR. Explicit roots supplied by trusted application code
    remain supported for tests/controlled deployments.
    """

    ENV_ROOTS = {
        "tutorials": "FOAM_TUTORIALS",
        "source": "FOAM_SRC",
        "etc": "FOAM_ETC",
        "modules": "FOAM_MODULES",
    }

    def __init__(self, roots: dict[str, str | Path] | None = None):
        self.roots: dict[str, Path] = {}
        self.last_search_metadata = {}
        self.last_read_metadata = {}
        self._content_cache = OrderedDict()
        if roots is not None:
            for scope, value in roots.items():
                if scope not in self.ENV_ROOTS or not value:
                    continue
                path = Path(value).expanduser().resolve()
                if path.is_dir():
                    self.roots[scope] = path
            return

        project_text = os.environ.get("WM_PROJECT_DIR", "").strip()
        if not project_text:
            return
        project_root = Path(project_text).expanduser().resolve()
        if not project_root.is_dir():
            return
        for scope, env_name in self.ENV_ROOTS.items():
            value = os.environ.get(env_name, "").strip()
            if not value:
                continue
            path = Path(value).expanduser().resolve()
            if path.is_dir() and _is_within(path, project_root):
                self.roots[scope] = path

    def summary(self) -> dict[str, dict[str, object]]:
        # Never reveal absolute installation paths to the model.
        return {
            scope: {"available": True, "reference_prefix": f"{scope}:"}
            for scope in sorted(self.roots)
        }

    def search(
        self,
        query: str,
        *,
        scope: str = "all",
        limit: int = 12,
        max_files: int = 6000,
    ) -> list[dict[str, object]]:
        if not 1 <= max_files <= 100000 or not 1 <= limit <= 100:
            raise ValueError("Reference scan bounds are outside policy.")
        query_tokens = normalize_query(query)
        selected = self._selected_roots(scope)
        inspected = {name: 0 for name, _ in selected}
        exhausted = set()
        iterators = {name: iter(_safe_files(root)) for name, root in selected}
        roots = dict(selected)
        results = []
        total = 0
        # Round-robin scopes: a large source tree cannot starve tutorials/modules.
        while total < max_files and len(exhausted) < len(selected):
            for name, root in selected:
                if name in exhausted or total >= max_files:
                    continue
                path = next(iterators[name], None)
                if path is None:
                    exhausted.add(name)
                    continue
                total += 1
                inspected[name] += 1
                relative = path.relative_to(root).as_posix()
                name_score = sum(4 for token in query_tokens if token in relative.casefold())
                try:
                    stat = path.stat()
                    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino)
                    text = self._content_cache.get(key)
                    if text is None:
                        text = path.read_text(encoding="utf-8", errors="replace") if stat.st_size <= 1_000_000 else ""
                        self._content_cache[key] = text
                        while len(self._content_cache) > 32:
                            self._content_cache.popitem(last=False)
                except OSError:
                    continue
                lowered = text.casefold()
                score = name_score + sum(token in lowered for token in query_tokens)
                if score:
                    first = min((lowered.find(t) for t in query_tokens if t in lowered), default=0)
                    snippet = text[max(0, first-160):first+500]
                    reference = f"{name}:{relative}"
                    results.append((score,reference,{"reference": reference, "scope": name,
                        "path": relative, "snippet": snippet,
                        "excerpt_sha256": hashlib.sha256(snippet.encode()).hexdigest()}))
        self.last_search_metadata = {"normalized_tokens": query_tokens, "inspected_by_scope": inspected,
            "exhausted_scopes": sorted(exhausted), "possibly_truncated_scopes": sorted(set(roots)-exhausted),
            "uninspected_scopes": sorted(name for name,count in inspected.items() if count == 0),
            "file_budget": max_files, "cache_scope": "trusted root/path/size/mtime/inode; at most 32 file bodies"}
        results.sort(key=lambda item: (-item[0],item[1]))
        return [item[2] for item in results[:limit]]

    def read(self, reference: str, *, start_line: int = 1, line_count: int = 160) -> str:
        scope, separator, relative_text = reference.partition(":")
        if not separator or scope not in self.roots:
            raise ValueError(f"Unknown OpenFOAM reference: {reference}")
        root = self.roots[scope]
        relative = Path(relative_text)
        path = (root / relative).resolve()
        if relative.is_absolute() or not _is_within(path, root) or path == root or not path.is_file():
            raise ValueError(f"Reference escapes the installed {scope} root: {reference}")
        if not 1 <= start_line <= 1000000 or not 1 <= line_count <= 400:
            raise ValueError("Reference line selection exceeds bounds.")
        if any(part.is_symlink() for part in [root / relative, *(root / relative).parents] if part != root):
            raise ValueError("Symbolic links are not accepted as source evidence.")
        if path.stat().st_size > 10_000_000:
            raise ValueError("Reference file exceeds the bounded read limit.")
        digest = hashlib.sha256()
        selected = []
        total_chars = 0
        with path.open("rb") as stream:
            for index, raw in enumerate(stream, start=1):
                digest.update(raw)
                if start_line <= index < start_line+line_count:
                    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    total_chars += len(text)
                    if total_chars > 120000:
                        raise ValueError("Reference excerpt exceeds its character budget; choose a smaller line range.")
                    selected.append(f"{index}: {text}")
        excerpt = "\n".join(selected)
        self.last_read_metadata = {"reference": reference, "source_sha256": digest.hexdigest(),
            "excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
            "start_line": start_line, "lines_returned": len(selected), "source_size_bytes": path.stat().st_size}
        return excerpt

    def _selected_roots(self, scope: str) -> list[tuple[str, Path]]:
        if scope == "all":
            return sorted(self.roots.items())
        if scope not in self.ENV_ROOTS:
            raise ValueError(f"Unsupported reference scope: {scope}")
        path = self.roots.get(scope)
        return [(scope, path)] if path is not None else []


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _safe_files(root):
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base)/d).is_symlink())
        for name in sorted(names):
            path = Path(base)/name
            if not path.is_symlink() and path.is_file() and _is_within(path.resolve(),root):
                yield path


def normalize_query(query):
    normalized = unicodedata.normalize("NFKC", query).casefold()
    aliases = {"\uaca9\uc790": "mesh blockMesh snappyHexMesh", "\uc5f4\uc804\ub2ec": "heat transfer thermo",
        "\uc628\ub3c4": "temperature", "\uc555\ub825": "pressure", "\uacbd\uacc4\uc870\uac74": "boundary conditions",
        "\ud6c4\ucc98\ub9ac": "functionObject postprocessing", "\uc720\ub7c9": "flowRate", "\ub2e4\uc911\uc601\uc5ed": "multi region",
        "\uc5f4\uc6d0": "heatSource", "\uc810\uc131": "viscosity", "\ub09c\ub958": "turbulence"}
    for source, target in aliases.items():
        if source in normalized:
            normalized += " " + target.casefold()
    return list(dict.fromkeys(token for token in re.findall(r"[\w.+-]+",normalized,re.UNICODE) if token))
