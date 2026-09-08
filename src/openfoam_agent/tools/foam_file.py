from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from typing import Iterable

from openfoam_agent.schemas.engineering import FoamDictionaryEntry


_FIELD_CLASS_RE = re.compile(
    r"^(?:vol|surface)(?:Scalar|Vector|SphericalTensor|SymmTensor|Tensor)Field$"
)
_CLASS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_HEADER_VALUE_RE = re.compile(r"\b(version|format|class|location|object)\s+([^;{}]+);", re.DOTALL)
_NONUNIFORM_RE = re.compile(
    r"\bnonuniform\s+(?:List<)?(scalar|vector|sphericalTensor|symmTensor|tensor)>?\b",
    re.IGNORECASE,
)
_UNIFORM_RE = re.compile(r"^\s*uniform\s+(.+?)\s*$", re.DOTALL)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class FoamFileContractError(ValueError):
    """Raised when a case file cannot satisfy a deterministic FoamFile contract."""


@dataclass(frozen=True)
class FoamFileContract:
    path: str
    class_name: str
    object_name: str
    location: str
    format: str = "ascii"
    version: str = "2.0"


@dataclass(frozen=True)
class FoamFileHeader:
    class_name: str = ""
    object_name: str = ""
    location: str = ""
    format: str = ""
    version: str = ""
    present: bool = False
    complete: bool = False


@dataclass(frozen=True)
class FoamFileHeaderValidation:
    valid: bool
    path: str
    header: FoamFileHeader
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        if self.valid:
            return (
                f"FoamFile header OK: {self.path} "
                f"class={self.header.class_name} object={self.header.object_name}"
            )
        return "\n".join(self.failures)


def is_field_class(class_name: str) -> bool:
    return bool(_FIELD_CLASS_RE.fullmatch(class_name.strip()))


def resolve_foam_file_contract(
    path: str,
    *,
    entries: Iterable[FoamDictionaryEntry] = (),
    explicit_class: str | None = None,
) -> tuple[FoamFileContract, list[FoamDictionaryEntry]]:
    """Resolve canonical FoamFile metadata and strip legacy FoamFile.* leaf entries.

    The path owns object/location. The serializer owns format/version. For ordinary
    dictionaries the class is deterministic. Initial-field class is either supplied
    explicitly or inferred only when the internalField expression is unambiguous.
    Legacy FoamFile leaf entries are accepted only when they agree with the canonical
    contract; they never render a second, Agent-authored header.
    """

    normalized_path = _normalize_case_path(path)
    root = normalized_path.split("/", 1)[0]
    object_name = PurePosixPath(normalized_path).name
    location = str(PurePosixPath(normalized_path).parent)
    if location == ".":
        location = ""

    body_entries: list[FoamDictionaryEntry] = []
    legacy: dict[str, str] = {}
    for entry in entries:
        if entry.path == "FoamFile" or entry.path.startswith("FoamFile."):
            if entry.path == "FoamFile":
                marker = " ".join(entry.value.strip().lower().split())
                if marker not in {"{}", "{ }", "{\n}", "block", "dictionary"}:
                    raise FoamFileContractError(
                        "Typed dictionaries must not author a raw FoamFile block; "
                        "Python owns the canonical OpenFOAM file header."
                    )
                continue
            key = entry.path.split(".", 1)[1]
            if key not in {"version", "format", "class", "location", "object"}:
                raise FoamFileContractError(
                    f"Unsupported legacy FoamFile metadata key {entry.path!r}; "
                    "Python owns the canonical header."
                )
            legacy[key] = _clean_header_value(entry.value)
            continue
        body_entries.append(entry)

    declared_class = (explicit_class or "").strip() or legacy.get("class", "")
    if declared_class and not _CLASS_RE.fullmatch(declared_class):
        raise FoamFileContractError(f"Unsafe OpenFOAM class name: {declared_class!r}")

    if root == "0":
        inferred_class = infer_field_class_from_entries(body_entries)
        if declared_class:
            class_name = declared_class
            if not is_field_class(class_name):
                raise FoamFileContractError(
                    f"Initial field {normalized_path!r} requires a field FoamFile class; "
                    f"got {class_name!r}."
                )
            if inferred_class and inferred_class != class_name:
                raise FoamFileContractError(
                    f"FoamFile class {class_name!r} conflicts with unambiguous internalField "
                    f"shape {inferred_class!r} for {normalized_path}."
                )
        elif inferred_class:
            class_name = inferred_class
        else:
            raise FoamFileContractError(
                f"Cannot prove the FoamFile class for initial field {normalized_path!r}. "
                "Set foam_class explicitly (for example volScalarField or volVectorField)."
            )
    else:
        # Generic typed dictionaries are dictionaries unless the Agent explicitly
        # declares a specialized OpenFOAM object class (e.g. a constant field).
        class_name = declared_class or "dictionary"

    if legacy.get("object") and legacy["object"] != object_name:
        raise FoamFileContractError(
            f"FoamFile object {legacy['object']!r} conflicts with path-derived object "
            f"{object_name!r} for {normalized_path}."
        )
    if legacy.get("location") and legacy["location"] != location:
        raise FoamFileContractError(
            f"FoamFile location {legacy['location']!r} conflicts with path-derived location "
            f"{location!r} for {normalized_path}."
        )
    if legacy.get("format") and legacy["format"] != "ascii":
        raise FoamFileContractError(
            f"Typed text serializer only supports FoamFile format ascii; got {legacy['format']!r}."
        )
    if legacy.get("class") and legacy["class"] != class_name:
        raise FoamFileContractError(
            f"Legacy FoamFile class {legacy['class']!r} conflicts with resolved class {class_name!r}."
        )

    return (
        FoamFileContract(
            path=normalized_path,
            class_name=class_name,
            object_name=object_name,
            location=location,
        ),
        body_entries,
    )


def render_foam_file_header(contract: FoamFileContract) -> str:
    location_line = f'    location "{contract.location}";\n' if contract.location else ""
    return (
        "FoamFile\n"
        "{\n"
        f"    version {contract.version};\n"
        f"    format {contract.format};\n"
        f"    class {contract.class_name};\n"
        f"{location_line}"
        f"    object {contract.object_name};\n"
        "}\n"
    )


def validate_foam_file_header(
    path: str,
    text: str,
    *,
    expected_class: str | None = None,
    require_field_class: bool | None = None,
) -> FoamFileHeaderValidation:
    """Validate the native IOobject-facing FoamFile contract of one case file.

    This deliberately checks semantics that ``foamDictionary -keywords`` does not
    prove: a top-level FoamFile block, required format/class/object metadata, path/object
    consistency, and class constraints when the caller can prove them.
    """

    normalized_path = _normalize_case_path(path)
    clean = _strip_comments(text)
    start = _first_non_ws(clean)
    failures: list[str] = []
    warnings: list[str] = []

    if start is None or not clean.startswith("FoamFile", start):
        header = FoamFileHeader(present=False, complete=False)
        failures.append(
            f"OpenFOAM file header missing in {normalized_path}: first structural entry must be FoamFile."
        )
        return FoamFileHeaderValidation(False, normalized_path, header, tuple(failures), tuple(warnings))

    brace = clean.find("{", start + len("FoamFile"))
    if brace < 0:
        header = FoamFileHeader(present=True, complete=False)
        failures.append(f"Malformed FoamFile header in {normalized_path}: opening '{{' is missing.")
        return FoamFileHeaderValidation(False, normalized_path, header, tuple(failures), tuple(warnings))
    end = _matching_brace(clean, brace)
    if end is None:
        header = FoamFileHeader(present=True, complete=False)
        failures.append(f"Malformed FoamFile header in {normalized_path}: closing '}}' is missing.")
        return FoamFileHeaderValidation(False, normalized_path, header, tuple(failures), tuple(warnings))

    body = clean[brace + 1 : end]
    values: dict[str, str] = {}
    for match in _HEADER_VALUE_RE.finditer(body):
        values[match.group(1)] = _clean_header_value(match.group(2))

    header = FoamFileHeader(
        class_name=values.get("class", ""),
        object_name=values.get("object", ""),
        location=values.get("location", ""),
        format=values.get("format", ""),
        version=values.get("version", ""),
        present=True,
        complete=all(values.get(key) for key in ("format", "class", "object")),
    )

    for key in ("format", "class", "object"):
        if not values.get(key):
            failures.append(f"FoamFile header in {normalized_path} is missing required key {key!r}.")
    if header.format and header.format not in {"ascii", "binary"}:
        failures.append(
            f"FoamFile header in {normalized_path} has unsupported format {header.format!r}."
        )

    expected_object = PurePosixPath(normalized_path).name
    if header.object_name and header.object_name != expected_object:
        failures.append(
            f"FoamFile object mismatch in {normalized_path}: header={header.object_name!r}, "
            f"path object={expected_object!r}."
        )

    expected_location = str(PurePosixPath(normalized_path).parent)
    if expected_location == ".":
        expected_location = ""
    if header.location and header.location != expected_location:
        failures.append(
            f"FoamFile location mismatch in {normalized_path}: header={header.location!r}, "
            f"path location={expected_location!r}."
        )

    if expected_class and header.class_name and header.class_name != expected_class:
        failures.append(
            f"FoamFile class mismatch in {normalized_path}: header={header.class_name!r}, "
            f"expected={expected_class!r}."
        )

    if require_field_class is None:
        require_field_class = normalized_path.startswith("0/")
    if require_field_class and header.class_name and not is_field_class(header.class_name):
        failures.append(
            f"Initial field {normalized_path} has non-field FoamFile class {header.class_name!r}."
        )

    if normalized_path.startswith("0/") and header.class_name:
        inferred = infer_field_class_from_text(text)
        if inferred and inferred != header.class_name:
            failures.append(
                f"FoamFile class/internalField mismatch in {normalized_path}: "
                f"header={header.class_name!r}, inferred={inferred!r}."
            )

    if not header.version:
        warnings.append(f"FoamFile header in {normalized_path} omits optional version metadata.")
    if not header.location:
        warnings.append(f"FoamFile header in {normalized_path} omits optional location metadata.")

    return FoamFileHeaderValidation(
        valid=not failures,
        path=normalized_path,
        header=header,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def infer_field_class_from_entries(entries: Iterable[FoamDictionaryEntry]) -> str | None:
    for entry in entries:
        if entry.path == "internalField":
            return infer_field_class_from_value(entry.value)
    return None


def infer_field_class_from_text(text: str) -> str | None:
    clean = _strip_comments(text)
    match = re.search(r"\binternalField\s+(.+?);", clean, re.DOTALL)
    if match is None:
        return None
    return infer_field_class_from_value(match.group(1))


def infer_field_class_from_value(value: str) -> str | None:
    rendered = value.strip().rstrip(";").strip()
    nonuniform = _NONUNIFORM_RE.search(rendered)
    if nonuniform:
        return _value_kind_to_vol_class(nonuniform.group(1))

    uniform = _UNIFORM_RE.match(rendered)
    if uniform is None:
        return None
    payload = uniform.group(1).strip()
    if _NUMBER_RE.fullmatch(payload):
        return "volScalarField"
    if payload.startswith("(") and payload.endswith(")"):
        atoms = [part for part in re.split(r"\s+", payload[1:-1].strip()) if part]
        if not atoms or not all(_NUMBER_RE.fullmatch(part) for part in atoms):
            return None
        return {
            1: "volSphericalTensorField",
            3: "volVectorField",
            6: "volSymmTensorField",
            9: "volTensorField",
        }.get(len(atoms))
    return None


def _value_kind_to_vol_class(kind: str) -> str | None:
    key = kind.casefold()
    return {
        "scalar": "volScalarField",
        "vector": "volVectorField",
        "sphericaltensor": "volSphericalTensorField",
        "symmtensor": "volSymmTensorField",
        "tensor": "volTensorField",
    }.get(key)


def _normalize_case_path(path: str) -> str:
    value = path.strip().replace("\\", "/")
    if not value or value.startswith("/") or ".." in PurePosixPath(value).parts:
        raise FoamFileContractError(f"Unsafe OpenFOAM case path: {path!r}")
    return str(PurePosixPath(value))


def _clean_header_value(value: str) -> str:
    rendered = value.strip().rstrip(";").strip()
    if len(rendered) >= 2 and rendered[0] == rendered[-1] and rendered[0] in {'"', "'"}:
        rendered = rendered[1:-1]
    return rendered.strip()


def _strip_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if quote is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in {'"', "'"}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i = min(len(text), i + 2)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _first_non_ws(text: str) -> int | None:
    for index, char in enumerate(text):
        if not char.isspace():
            return index
    return None


def _matching_brace(text: str, start: int) -> int | None:
    depth = 0
    quote: str | None = None
    i = start
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None
