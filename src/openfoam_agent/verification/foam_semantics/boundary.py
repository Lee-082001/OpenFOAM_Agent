from __future__ import annotations

import re

from .model import (
    BoundaryMatchKind,
    BoundaryResolution,
    BoundarySelector,
    MeshIR,
    MeshPatch,
    PatternState,
    ResolutionStatus,
    ResolutionTraceStep,
)
from .parser import find_named_dictionary, parse_top_level_blocks


def parse_boundary_selectors(text: str) -> tuple[BoundarySelector, ...]:
    start = find_named_dictionary(text, "boundaryField")
    if start is None:
        return ()
    section = parse_top_level_blocks(text, start, "}")
    return tuple(
        BoundarySelector(
            key=entry.key,
            order=entry.order,
            dictionary_body=entry.body,
            field_type=entry.declared_type,
        )
        for entry in section.entries
    )


class BoundaryFieldInterpreter:
    """Resolve field boundary entries using OpenFOAM v13 selection semantics.

    The deterministic tiers model GeometricBoundaryField::readField(): explicit
    patch names, patch groups, automatic empty patch fields, then wildcard/wordRe
    patterns. Dynamic constructs that Python cannot prove are retained as
    INDETERMINATE instead of being misreported as missing.
    """

    def resolve_all(self, mesh: MeshIR, selectors: tuple[BoundarySelector, ...]) -> dict[str, BoundaryResolution]:
        return {patch.name: self.resolve(patch, selectors) for patch in mesh.patches}

    def resolve(self, patch: MeshPatch, selectors: tuple[BoundarySelector, ...]) -> BoundaryResolution:
        trace: list[ResolutionTraceStep] = []
        literal = [s for s in selectors if s.key.pattern_state == PatternState.LITERAL]
        patterns = [s for s in selectors if s.key.pattern_state == PatternState.PATTERN]
        unknown = [s for s in selectors if s.key.pattern_state == PatternState.INDETERMINATE]

        # Dictionary expansions/directives can inject selectors at stronger tiers.
        # Until expanded by OpenFOAM itself, Python cannot prove which entry is
        # effective, even when a visible fallback would otherwise match.
        if unknown:
            reason = "unresolved dynamic selector(s): " + ", ".join(item.key.raw for item in unknown)
            trace.append(ResolutionTraceStep("indeterminate", reason))
            return BoundaryResolution(
                patch=patch,
                status=ResolutionStatus.INDETERMINATE,
                match_kind=BoundaryMatchKind.INDETERMINATE,
                selector=None,
                effective_field_type="",
                certainty="indeterminate",
                trace=tuple(trace),
                reason=reason,
            )

        exact = [s for s in literal if s.key.value == patch.name]
        if exact:
            selected = exact[-1]
            trace.append(ResolutionTraceStep("exact", f'matched {selected.key.raw}'))
            return _resolved(patch, BoundaryMatchKind.EXACT, selected, trace)
        trace.append(ResolutionTraceStep("exact", "no explicit patch entry matched"))

        group = [s for s in literal if s.key.value in patch.groups]
        if group:
            # OpenFOAM resolves overlapping groups in reverse dictionary order.
            selected = max(group, key=lambda item: item.order)
            trace.append(ResolutionTraceStep("group", f'matched group {selected.key.raw}'))
            return _resolved(patch, BoundaryMatchKind.GROUP, selected, trace)
        trace.append(ResolutionTraceStep("group", "no patchGroup entry matched"))

        if patch.patch_type == "empty":
            trace.append(ResolutionTraceStep("auto_empty", "mesh patch type is empty; OpenFOAM auto-empty fallback applies"))
            return BoundaryResolution(
                patch=patch,
                status=ResolutionStatus.RESOLVED,
                match_kind=BoundaryMatchKind.AUTO_EMPTY,
                selector=None,
                effective_field_type="empty",
                certainty="proven",
                trace=tuple(trace),
            )
        trace.append(ResolutionTraceStep("auto_empty", "not applicable"))

        matches: list[BoundarySelector] = []
        regex_errors: list[str] = []
        for selector in patterns:
            try:
                if re.fullmatch(selector.key.value, patch.name):
                    matches.append(selector)
            except re.error as exc:
                regex_errors.append(f"{selector.key.raw}: {exc}")
        if matches:
            selected = max(matches, key=lambda item: item.order)
            trace.append(ResolutionTraceStep("regex", f'matched {selected.key.raw}; later matching pattern wins'))
            return _resolved(patch, BoundaryMatchKind.REGEX, selected, trace)
        trace.append(ResolutionTraceStep("regex", "no supported pattern matched"))

        if regex_errors:
            reason = "pattern(s) not safely interpreted by Python: " + "; ".join(regex_errors)
            trace.append(ResolutionTraceStep("indeterminate", reason))
            return BoundaryResolution(
                patch=patch,
                status=ResolutionStatus.INDETERMINATE,
                match_kind=BoundaryMatchKind.INDETERMINATE,
                selector=None,
                effective_field_type="",
                certainty="indeterminate",
                trace=tuple(trace),
                reason=reason,
            )

        trace.append(ResolutionTraceStep("missing", "no selector can cover this patch"))
        return BoundaryResolution(
            patch=patch,
            status=ResolutionStatus.MISSING,
            match_kind=BoundaryMatchKind.NONE,
            selector=None,
            effective_field_type="",
            certainty="proven",
            trace=tuple(trace),
            reason="no boundaryField selector matched",
        )


def _resolved(
    patch: MeshPatch,
    kind: BoundaryMatchKind,
    selector: BoundarySelector,
    trace: list[ResolutionTraceStep],
) -> BoundaryResolution:
    return BoundaryResolution(
        patch=patch,
        status=ResolutionStatus.RESOLVED,
        match_kind=kind,
        selector=selector,
        effective_field_type=selector.field_type,
        certainty="proven",
        trace=tuple(trace),
    )
