from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from openfoam_agent.schemas.engineering import CaseFileSeal, CaseSeal, EngineeringPlan


_WRITABLE_TOP_LEVEL = {"0", "constant", "system", "postprocessConfig"}
_EXECUTION_INPUT_TOP_LEVEL = {"0", "constant", "system"}
_MESH_AFFECTING_EXACT_PATHS = {
    "system/blockMeshDict",
    "system/surfaceFeatureExtractDict",
    "system/snappyHexMeshDict",
    "system/createPatchDict",
}
_MESH_AFFECTING_PREFIXES = (
    "constant/polyMesh/",
    "constant/triSurface/",
)
_MESH_GEOMETRY_SUFFIXES = {
    ".stl",
    ".obj",
    ".emesh",
    ".vtk",
    ".vtp",
    ".off",
    ".nas",
    ".bdf",
}
_TIME_DIR = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
_SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_FORBIDDEN_CONTENT = (
    "#codestream",
    "#calc",
    "#include",
    "#includeetc",
    "dynamiccode",
    "codedfixedvalue",
    "codemixed",
    "codedfunctionobject",
    "codedsource",
    "systemcall",
    "functionentry::",
    "codeinclude",
    "codeoptions",
    "codelibs",
)
_LIBS_ENTRY = re.compile(r"(?ms)^\s*libs\s*\((?P<body>.*?)\)\s*;")
_QUOTED_LIB = re.compile(r'["\'](?P<name>lib[A-Za-z][A-Za-z0-9_]*\.so)["\']')


class WorkspaceSafetyError(RuntimeError):
    pass


class CaseWorkspace:
    """Sandboxed, hashable agent workspace for one case.

    Agent-visible paths are always relative to ``case/`` and restricted to the
    conventional OpenFOAM data roots. Tool logs live outside the agent-authored
    tree so they cannot be confused with sealed inputs.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = 1_000_000,
        max_total_bytes: int = 8_000_000,
        max_execution_bytes: int = 2_000_000_000,
        max_execution_files: int = 50_000,
        allowed_libraries: set[str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.case_dir = self.root / "case"
        self.log_dir = self.root / "logs"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Case inputs and logs can contain proprietary geometry/BC details. Keep the
        # run workspace private even when the parent is a shared /tmp directory.
        for private_dir in (self.root, self.case_dir, self.log_dir):
            os.chmod(private_dir, 0o700)
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_execution_bytes = max_execution_bytes
        self.max_execution_files = max_execution_files
        self.allowed_libraries = allowed_libraries or {
            "libfvMotionSolvers.so",
            "librigidBodyMeshMotion.so",
            "libforces.so",
            "libfieldFunctionObjects.so",
        }
        self._authored_paths: set[str] = set()

    def resolve_case_path(self, relative_text: str, *, must_exist: bool = False) -> Path:
        relative = self._validate_relative(relative_text)
        path = (self.case_dir / relative).resolve()
        if self.case_dir not in path.parents:
            raise WorkspaceSafetyError(f"Case path escapes sandbox: {relative_text}")
        if must_exist and not path.is_file():
            raise FileNotFoundError(f"Case file does not exist: {relative_text}")
        return path

    def write_text(self, relative_text: str, content: str) -> str:
        if "\x00" in content:
            raise WorkspaceSafetyError("Case files must not contain NUL bytes.")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise WorkspaceSafetyError(
                f"Case file exceeds {self.max_file_bytes} byte limit: {relative_text}"
            )
        self._validate_content(content, relative_text)
        path = self.resolve_case_path(relative_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, content)
        self._authored_paths.add(self._normalized(relative_text))
        self._assert_total_size()
        return hashlib.sha256(encoded).hexdigest()

    def read_text(self, relative_text: str, *, max_chars: int = 40_000) -> str:
        path = self.resolve_case_path(relative_text, must_exist=True)
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) > max_chars:
            return data[:max_chars] + "\n... [truncated]"
        return data

    def case_file_digest(self, relative_text: str) -> str:
        path = self.resolve_case_path(relative_text, must_exist=True)
        return _sha256_file(path)

    def delete(self, relative_text: str) -> None:
        normalized = self._normalized(relative_text)
        path = self.resolve_case_path(normalized, must_exist=True)
        path.unlink()
        self._authored_paths.discard(normalized)
        parent = path.parent
        while parent != self.case_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    def list_authored(self) -> list[str]:
        return sorted(path for path in self._authored_paths if self.resolve_case_path(path).is_file())

    def validate_all_content(self) -> list[str]:
        failures: list[str] = []
        for relative in self.list_authored():
            path = self.resolve_case_path(relative, must_exist=True)
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures.append(f"{relative}: agent-authored case inputs must be UTF-8 text")
                continue
            try:
                self._validate_content(content, relative)
            except WorkspaceSafetyError as exc:
                failures.append(str(exc))
        try:
            self._assert_total_size()
        except WorkspaceSafetyError as exc:
            failures.append(str(exc))
        return failures

    def manifest_digest(self) -> str:
        """Digest every execution input under 0/, constant/ and system/."""
        return _seal_manifest_digest(self.execution_file_seals())

    def mesh_manifest_digest(self) -> str:
        """Digest only artifacts whose content can affect the generated mesh.

        This is intentionally narrower than :meth:`manifest_digest`. Solver-control
        dictionaries and initial fields remain part of the full case seal, but changing
        them does not make previously observed checkMesh evidence stale.
        """
        mesh_files = [
            item
            for item in self.execution_file_seals()
            if self.is_mesh_affecting_path(item.path)
        ]
        return _seal_manifest_digest(mesh_files)

    def is_mesh_affecting_path(self, relative_text: str) -> bool:
        """Return whether a case path participates in the allowlisted mesh pipeline."""
        normalized = self._normalized(relative_text)
        if normalized in _MESH_AFFECTING_EXACT_PATHS:
            return True
        if normalized.startswith(_MESH_AFFECTING_PREFIXES):
            return True
        if normalized.startswith("constant/"):
            return Path(normalized).suffix.casefold() in _MESH_GEOMETRY_SUFFIXES
        return False

    def file_seals(self) -> list[CaseFileSeal]:
        """Return agent-authored files only, for agent observations/UI."""
        seals: list[CaseFileSeal] = []
        for relative in self.list_authored():
            path = self.resolve_case_path(relative, must_exist=True)
            data = path.read_bytes()
            seals.append(
                CaseFileSeal(
                    path=relative,
                    sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                    origin="agent",
                )
            )
        return seals

    def execution_file_seals(self) -> list[CaseFileSeal]:
        """Seal all pre-solve inputs, including mesh files created by native tools."""
        paths: list[Path] = []
        for top_level in sorted(_EXECUTION_INPUT_TOP_LEVEL):
            root = self.case_dir / top_level
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise WorkspaceSafetyError(f"Execution input must not be a symlink: {path}")
                if path.is_file():
                    paths.append(path)
        if len(paths) > self.max_execution_files:
            raise WorkspaceSafetyError(
                f"Case has {len(paths)} execution input files; limit is {self.max_execution_files}."
            )
        total = sum(path.stat().st_size for path in paths)
        if total > self.max_execution_bytes:
            raise WorkspaceSafetyError(
                f"Case execution inputs exceed {self.max_execution_bytes} byte limit."
            )
        seals: list[CaseFileSeal] = []
        for path in sorted(paths):
            relative = path.relative_to(self.case_dir).as_posix()
            self._validate_relative(relative)
            size = path.stat().st_size
            seals.append(
                CaseFileSeal(
                    path=relative,
                    sha256=_sha256_file(path),
                    size_bytes=size,
                    origin="agent" if relative in self._authored_paths else "native",
                )
            )
        return seals

    def seal(self, plan: EngineeringPlan) -> CaseSeal:
        if not self._authored_paths:
            raise WorkspaceSafetyError("Cannot seal an empty case workspace.")
        failures = self.validate_all_content()
        if failures:
            raise WorkspaceSafetyError("; ".join(failures))
        files = self.execution_file_seals()
        return CaseSeal(
            plan_sha256=plan.digest(),
            manifest_sha256=_seal_manifest_digest(files),
            files=files,
        )

    def adopt_seal(self, seal: CaseSeal) -> None:
        """Rehydrate agent-authored tracking while verifying sealed files exist."""
        authored: set[str] = set()
        for item in seal.files:
            self.resolve_case_path(item.path, must_exist=True)
            if item.origin == "agent":
                authored.add(item.path)
        self._authored_paths = authored

    def verify_seal(self, seal: CaseSeal, plan: EngineeringPlan) -> None:
        if plan.digest() != seal.plan_sha256:
            raise WorkspaceSafetyError("Engineering plan changed after approval/sealing.")
        expected = {item.path: item for item in seal.files}
        current = {item.path: item for item in self.execution_file_seals()}
        if set(expected) != set(current):
            raise WorkspaceSafetyError("Pre-solve execution input file set changed after sealing.")
        for path, item in expected.items():
            actual = current[path]
            if actual.sha256 != item.sha256 or actual.size_bytes != item.size_bytes:
                raise WorkspaceSafetyError(f"Sealed execution input changed: {path}")
            if actual.origin != item.origin:
                raise WorkspaceSafetyError(f"Execution input origin changed: {path}")
        current_files = [current[path] for path in sorted(current)]
        if _seal_manifest_digest(current_files) != seal.manifest_sha256:
            raise WorkspaceSafetyError("Case execution-input manifest changed after sealing.")

    def write_postprocess_config(self, relative_text: str, content: str) -> str:
        normalized = self._normalized(relative_text)
        if not normalized.startswith("postprocessConfig/"):
            raise WorkspaceSafetyError(
                "Post-processing configuration must live under postprocessConfig/."
            )
        return self.write_text(normalized, content)

    def resolve_result_path(self, relative_text: str, *, must_exist: bool = False) -> Path:
        """Resolve a read-only native result path without exposing arbitrary files.

        Result reads are limited to OpenFOAM time directories and postProcessing/.
        They are intentionally separate from agent-authored inputs so post-processing
        can inspect native outputs without expanding the write sandbox.
        """
        normalized = self._normalized_result(relative_text)
        relative = Path(normalized)
        path = (self.case_dir / relative).resolve()
        if self.case_dir not in path.parents:
            raise WorkspaceSafetyError(f"Result path escapes sandbox: {relative_text}")
        if path.is_symlink():
            raise WorkspaceSafetyError(f"Result path must not be a symlink: {relative_text}")
        if must_exist and not path.is_file():
            raise FileNotFoundError(f"Result file does not exist: {relative_text}")
        return path

    def list_result_files(self, prefix: str = "", *, max_files: int = 4000) -> list[dict[str, object]]:
        roots: list[Path] = []
        post = self.case_dir / "postProcessing"
        if post.is_dir():
            roots.append(post)
        for candidate in self.case_dir.iterdir():
            if candidate.is_dir() and _TIME_DIR.fullmatch(candidate.name):
                roots.append(candidate)

        normalized_prefix = prefix.strip("/")
        results: list[dict[str, object]] = []
        for root in sorted(roots):
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    continue
                if not path.is_file():
                    continue
                relative = path.relative_to(self.case_dir).as_posix()
                try:
                    self._normalized_result(relative)
                except WorkspaceSafetyError:
                    continue
                if normalized_prefix and not relative.startswith(normalized_prefix):
                    continue
                results.append({"path": relative, "size_bytes": path.stat().st_size})
                if len(results) >= max_files:
                    return results
        return results

    def read_result_text(self, relative_text: str, *, max_chars: int = 40_000) -> str:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        path = self.resolve_result_path(relative_text, must_exist=True)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = handle.read(max_chars + 1)
        if len(data) > max_chars:
            return data[:max_chars] + "\n... [truncated]"
        return data

    def result_file_digest(self, relative_text: str) -> tuple[str, int]:
        path = self.resolve_result_path(relative_text, must_exist=True)
        return _sha256_file(path), path.stat().st_size

    def archive_revision_outputs(self, revision_id: str) -> str:
        """Snapshot baseline inputs and isolate prior outputs before a confirmed revision.

        Baseline execution inputs are copied for rollback/audit. Prior runtime and
        post-processing outputs are moved out of the active case so stale evidence
        cannot contaminate the revised run. This is deterministic housekeeping, not
        a CFD modeling decision.
        """
        if not re.fullmatch(r"rev-[0-9]{4}", revision_id):
            raise WorkspaceSafetyError(f"Unsafe revision archive id: {revision_id!r}")
        archive_root = self.root / "revision-history" / revision_id
        if archive_root.exists():
            raise WorkspaceSafetyError(f"Revision archive already exists: {revision_id}")
        archive_root.mkdir(parents=True, exist_ok=False)
        os.chmod(archive_root.parent, 0o700)
        os.chmod(archive_root, 0o700)

        # Copy the exact pre-revision solver inputs first. Native tools may later
        # replace or mutate mesh files, so a hash-only record is not sufficient for
        # practical rollback/comparison. The execution-input size/file limits already
        # bound this snapshot.
        baseline_root = archive_root / "baseline_inputs"
        baseline_root.mkdir(parents=True, exist_ok=True)
        os.chmod(baseline_root, 0o700)
        for item in self.execution_file_seals():
            source = self.resolve_case_path(item.path, must_exist=True)
            destination = baseline_root / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(destination.parent, 0o700)
            shutil.copy2(source, destination)
            if _sha256_file(destination) != item.sha256 or destination.stat().st_size != item.size_bytes:
                raise WorkspaceSafetyError(
                    f"Revision baseline snapshot mismatch while copying sealed input: {item.path}"
                )

        candidates: list[tuple[Path, Path]] = []
        for item in self.case_dir.iterdir():
            if not item.is_dir() or item.is_symlink():
                continue
            if item.name == "0":
                continue
            if _TIME_DIR.fullmatch(item.name) or item.name in {"postProcessing", "postprocessConfig"}:
                candidates.append((item, archive_root / "case_outputs" / item.name))
        for item in self.log_dir.iterdir():
            if item.is_symlink():
                continue
            candidates.append((item, archive_root / "logs" / item.name))

        for source, destination in candidates:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(destination.parent, 0o700)
            shutil.move(str(source), str(destination))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.log_dir, 0o700)
        return archive_root.relative_to(self.root).as_posix()

    def write_log(self, name: str, content: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "tool"
        path = self.log_dir / safe
        self._atomic_write(path, content)
        return path

    def _validate_relative(self, relative_text: str) -> Path:
        normalized = self._normalized(relative_text)
        relative = Path(normalized)
        parts = relative.parts
        if (
            not parts
            or relative.is_absolute()
            or parts[0] not in _WRITABLE_TOP_LEVEL
            or any(part in {"", ".", ".."} or not _SAFE_PART.fullmatch(part) for part in parts)
        ):
            raise WorkspaceSafetyError(f"Unsafe case path: {relative_text!r}")
        return relative

    def _normalized_result(self, relative_text: str) -> str:
        normalized = self._normalized(relative_text)
        relative = Path(normalized)
        parts = relative.parts
        if not parts or relative.is_absolute():
            raise WorkspaceSafetyError(f"Unsafe result path: {relative_text!r}")
        if any(part in {"", ".", ".."} or not _SAFE_PART.fullmatch(part) for part in parts):
            raise WorkspaceSafetyError(f"Unsafe result path: {relative_text!r}")
        if parts[0] != "postProcessing" and not _TIME_DIR.fullmatch(parts[0]):
            raise WorkspaceSafetyError(f"Result path is outside native output roots: {relative_text!r}")
        return normalized

    @staticmethod
    def _normalized(relative_text: str) -> str:
        if "\\" in relative_text:
            raise WorkspaceSafetyError("Case paths must use forward slashes.")
        return relative_text.strip("/")

    def _validate_content(self, content: str, relative_text: str) -> None:
        lowered = content.casefold()
        forbidden = [token for token in _FORBIDDEN_CONTENT if token in lowered]
        if re.search(r"(?<![A-Za-z0-9_])system\s*\(", lowered):
            forbidden.append("system(...)")
        if re.search(r"(?<![A-Za-z0-9_])coded[A-Za-z0-9_]*\b", lowered):
            forbidden.append("coded* runtime code")
        if forbidden:
            raise WorkspaceSafetyError(
                f"{relative_text} contains executable/unsafe directives: "
                + ", ".join(sorted(set(forbidden)))
            )
        for matched in _LIBS_ENTRY.finditer(content):
            body = matched.group("body")
            libraries = {item.group("name") for item in _QUOTED_LIB.finditer(body)}
            residue = _QUOTED_LIB.sub("", body)
            if residue.strip():
                raise WorkspaceSafetyError(
                    f"{relative_text} has an unsupported libs entry syntax."
                )
            unknown = libraries - self.allowed_libraries
            if unknown:
                raise WorkspaceSafetyError(
                    f"{relative_text} requests non-allowlisted libraries: "
                    + ", ".join(sorted(unknown))
                )

    def _assert_total_size(self) -> None:
        total = sum(self.resolve_case_path(path, must_exist=True).stat().st_size for path in self.list_authored())
        if total > self.max_total_bytes:
            raise WorkspaceSafetyError(
                f"Agent-authored case exceeds {self.max_total_bytes} byte limit."
            )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_manifest_digest(files: list[CaseFileSeal]) -> str:
    records = [item.model_dump(mode="json") for item in sorted(files, key=lambda item: item.path)]
    canonical = json.dumps(records, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
