from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FoamTokenKind(str, Enum):
    WORD = "word"
    STRING = "string"
    DIRECTIVE = "directive"
    EXPANSION = "expansion"
    UNKNOWN = "unknown"


class PatternState(str, Enum):
    LITERAL = "literal"
    PATTERN = "pattern"
    INDETERMINATE = "indeterminate"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    MISSING = "missing"
    INDETERMINATE = "indeterminate"
    INVALID = "invalid"


class BoundaryMatchKind(str, Enum):
    EXACT = "exact"
    GROUP = "group"
    AUTO_EMPTY = "auto_empty"
    REGEX = "regex"
    INDETERMINATE = "indeterminate"
    NONE = "none"


@dataclass(frozen=True)
class FoamKey:
    raw: str
    value: str
    token_kind: FoamTokenKind
    pattern_state: PatternState


@dataclass(frozen=True)
class FoamEntry:
    key: FoamKey
    body: str
    order: int
    declared_type: str = ""


@dataclass(frozen=True)
class FoamDictionary:
    entries: tuple[FoamEntry, ...] = ()
    complete: bool = True


@dataclass(frozen=True)
class MeshPatch:
    name: str
    patch_type: str = ""
    groups: frozenset[str] = frozenset()
    order: int = 0


@dataclass(frozen=True)
class MeshIR:
    patches: tuple[MeshPatch, ...] = ()

    @property
    def names(self) -> list[str]:
        return [patch.name for patch in self.patches]

    @property
    def patch_types(self) -> dict[str, str]:
        return {patch.name: patch.patch_type for patch in self.patches}


@dataclass(frozen=True)
class BoundarySelector:
    key: FoamKey
    order: int
    dictionary_body: str
    field_type: str = ""


@dataclass(frozen=True)
class ResolutionTraceStep:
    stage: str
    outcome: str


@dataclass(frozen=True)
class BoundaryResolution:
    patch: MeshPatch
    status: ResolutionStatus
    match_kind: BoundaryMatchKind
    selector: BoundarySelector | None = None
    effective_field_type: str = ""
    certainty: str = "proven"
    trace: tuple[ResolutionTraceStep, ...] = field(default_factory=tuple)
    reason: str = ""
