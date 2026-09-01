from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from openfoam_agent.schemas.engineering import EngineeringPlan
from openfoam_agent.tools.openfoam import OpenFOAMTools
from openfoam_agent.tools.workspace import CaseWorkspace


_CORE_SYSTEM_FILES = ("system/controlDict", "system/fvSchemes", "system/fvSolution")
_FIELD_DIR = "0/"
_WORD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")


@dataclass
class PreSolveValidationResult:
    valid: bool
    failures: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    mesh_patches: list[str] = field(default_factory=list)


class PreSolveCompletenessGate:
    """Validate solver-input completeness without making CFD engineering choices.

    The Engineering Agent declares solver-specific required case files. Python only
    verifies that those declarations exist, parse as OpenFOAM dictionaries where
    applicable, and that declared initial fields cover every mesh boundary patch.
    """

    def __init__(self, tools: OpenFOAMTools, workspace: CaseWorkspace) -> None:
        self.tools = tools
        self.workspace = workspace

    def validate(self, plan: EngineeringPlan) -> PreSolveValidationResult:
        return self.validate_required_case_files(plan.required_case_files)

    def validate_required_case_files(
        self,
        required_case_files: list[str],
    ) -> PreSolveValidationResult:
        """Validate an Agent-declared solver-input file set without a full plan.

        This supports short engineering sequences such as write -> dictionary check ->
        pre-solve readiness while keeping the final EngineeringPlan validation intact.
        """

        failures: list[str] = []
        required = list(dict.fromkeys([*_CORE_SYSTEM_FILES, *required_case_files]))
        field_files = [item for item in required_case_files if item.startswith(_FIELD_DIR)]
        if not field_files:
            failures.append(
                "EngineeringPlan.required_case_files must declare the solver-required initial field files under 0/."
            )

        for relative in required:
            path = self.workspace.resolve_case_path(relative)
            if not path.is_file():
                failures.append(f"Required solve input is missing: {relative}")
                continue
            if self._should_dictionary_validate(path):
                result = self.tools.foam_dictionary_validate(path, cwd=self.workspace.case_dir)
                if not result.success:
                    excerpt = "\n".join(part for part in (result.stdout, result.stderr) if part)[-1200:]
                    failures.append(f"foamDictionary rejected required solve input {relative}: {excerpt}")

        boundary_path = self.workspace.resolve_case_path("constant/polyMesh/boundary")
        mesh_patches: list[str] = []
        if not boundary_path.is_file():
            failures.append("constant/polyMesh/boundary is missing; mesh patch coverage cannot be verified.")
        else:
            mesh_patches = _parse_boundary_patch_names(
                boundary_path.read_text(encoding="utf-8", errors="replace")
            )
            if not mesh_patches:
                failures.append("No mesh boundary patches could be parsed from constant/polyMesh/boundary.")

        if mesh_patches:
            for relative in field_files:
                path = self.workspace.resolve_case_path(relative)
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                field_patches = _parse_boundary_field_names(text)
                missing = sorted(set(mesh_patches) - set(field_patches))
                if missing:
                    failures.append(
                        f"Boundary coverage mismatch in {relative}; missing patchField entries: {missing}"
                    )
                if "internalField" not in text:
                    failures.append(f"Required initial field {relative} does not declare internalField.")
                if "dimensions" not in text:
                    failures.append(f"Required initial field {relative} does not declare dimensions.")

        return PreSolveValidationResult(
            valid=not failures,
            failures=failures,
            checked_files=required,
            mesh_patches=mesh_patches,
        )

    @staticmethod
    def _should_dictionary_validate(path: Path) -> bool:
        return path.suffix.lower() not in {".stl", ".obj", ".off", ".vtk", ".csv", ".dat", ".emesh"}


def _parse_boundary_patch_names(text: str) -> list[str]:
    start = _find_list_start_after_count(text)
    if start is None:
        return []
    return _top_level_dictionary_names(text, start, closing=")")


def _parse_boundary_field_names(text: str) -> list[str]:
    match = re.search(r"\bboundaryField\b", text)
    if match is None:
        return []
    brace = text.find("{", match.end())
    if brace < 0:
        return []
    return _top_level_dictionary_names(text, brace, closing="}")


def _find_list_start_after_count(text: str) -> int | None:
    clean = _strip_comments(text)
    match = re.search(r"\n\s*\d+\s*\n\s*\(", clean)
    if match is None:
        match = re.search(r"\b\d+\s*\(", clean)
    if match is None:
        return None
    return clean.find("(", match.start())


def _top_level_dictionary_names(text: str, open_index: int, *, closing: str) -> list[str]:
    clean = _strip_comments(text)
    if open_index >= len(clean):
        return []
    opener = clean[open_index]
    expected_open = "(" if closing == ")" else "{"
    if opener != expected_open:
        # Callers may have computed an index against the unstripped string. Re-find
        # the corresponding section conservatively.
        open_index = clean.find(expected_open, max(0, open_index - 32))
        if open_index < 0:
            return []
    names: list[str] = []
    depth = 0
    token = ""
    i = open_index + 1
    while i < len(clean):
        ch = clean[i]
        if ch == expected_open:
            depth += 1
        elif ch == closing:
            if depth == 0:
                break
            depth -= 1
        if depth == 0:
            if ch.isalnum() or ch in "_.:-":
                token += ch
            else:
                if token and _WORD.match(token):
                    j = i
                    while j < len(clean) and clean[j].isspace():
                        j += 1
                    if j < len(clean) and clean[j] == "{":
                        names.append(token)
                token = ""
        i += 1
    return list(dict.fromkeys(names))


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
