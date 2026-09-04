from __future__ import annotations

import os
import re
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
    }

    def __init__(self, roots: dict[str, str | Path] | None = None):
        self.roots: dict[str, Path] = {}
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
        query_tokens = [
            token for token in re.findall(r"[A-Za-z0-9_.+-]+", query.casefold()) if token
        ]
        if not query_tokens:
            return []
        selected = self._selected_roots(scope)
        results: list[tuple[int, str, dict[str, object]]] = []
        inspected = 0
        for root_scope, root in selected:
            try:
                iterator = root.rglob("*")
                for path in iterator:
                    if inspected >= max_files:
                        break
                    try:
                        if not path.is_file() or path.is_symlink():
                            continue
                    except OSError:
                        continue
                    inspected += 1
                    try:
                        resolved = path.resolve()
                    except OSError:
                        continue
                    if not _is_within(resolved, root) or resolved == root:
                        continue
                    relative = resolved.relative_to(root).as_posix()
                    name_haystack = relative.casefold()
                    name_score = sum(4 for token in query_tokens if token in name_haystack)
                    content_score = 0
                    snippet = ""
                    text = ""
                    try:
                        size = resolved.stat().st_size
                    except OSError:
                        size = 1_000_001
                    if name_score == 0 and size <= 1_000_000:
                        try:
                            text = resolved.read_text(encoding="utf-8", errors="ignore")
                        except OSError:
                            text = ""
                        lowered = text.casefold()
                        content_score = sum(1 for token in query_tokens if token in lowered)
                        if content_score:
                            first = min(
                                (lowered.find(token) for token in query_tokens if token in lowered),
                                default=0,
                            )
                            start_pos = max(0, first - 160)
                            snippet = " ".join(text[start_pos:first + 500].split())[:700]
                    score = name_score + content_score
                    if score:
                        reference = f"{root_scope}:{relative}"
                        results.append(
                            (
                                score,
                                reference,
                                {
                                    "reference": reference,
                                    "scope": root_scope,
                                    "path": relative,
                                    "snippet": snippet,
                                },
                            )
                        )
            except OSError:
                # One unreadable installed subtree must not disable all capability/reference
                # retrieval. Other scopes remain usable and deterministic.
                continue
            if inspected >= max_files:
                break
        results.sort(key=lambda item: (-item[0], item[1]))
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
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = start_line - 1
        selected = lines[start:start + line_count]
        return "\n".join(
            f"{index}: {line}" for index, line in enumerate(selected, start=start_line)
        )

    def _selected_roots(self, scope: str) -> list[tuple[str, Path]]:
        if scope == "all":
            return sorted(self.roots.items())
        if scope not in self.ENV_ROOTS:
            raise ValueError(f"Unsupported reference scope: {scope}")
        path = self.roots.get(scope)
        return [(scope, path)] if path is not None else []


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
