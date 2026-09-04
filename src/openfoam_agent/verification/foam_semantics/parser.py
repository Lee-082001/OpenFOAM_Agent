from __future__ import annotations

import re
from .model import FoamDictionary, FoamEntry, FoamKey, FoamTokenKind, PatternState


_PATTERN_META = re.compile(r"[.*+?()|\[\]{}^$\\]")
_TYPE_RE = re.compile(r"\btype\s+([^;{}]+);")
_GROUPS_RE = re.compile(r"\binGroups\s+(?:\d+\s*)?\((.*?)\)\s*;", re.DOTALL)


def strip_comments(text: str) -> str:
    """Remove comments while preserving quoted strings and line structure."""
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


def parse_key(text: str, index: int) -> tuple[FoamKey | None, int]:
    i = _skip_ws(text, index)
    if i >= len(text):
        return None, i
    if text[i] in {'"', "'"}:
        quote = text[i]
        start = i
        i += 1
        value_chars: list[str] = []
        while i < len(text):
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                value_chars.extend((ch, text[i + 1]))
                i += 2
                continue
            if ch == quote:
                raw = text[start : i + 1]
                value = "".join(value_chars)
                state = PatternState.PATTERN if _PATTERN_META.search(value) else PatternState.LITERAL
                return FoamKey(raw, value, FoamTokenKind.STRING, state), i + 1
            value_chars.append(ch)
            i += 1
        raw = text[start:i]
        return FoamKey(raw, raw, FoamTokenKind.UNKNOWN, PatternState.INDETERMINATE), i

    start = i
    while i < len(text) and not text[i].isspace() and text[i] not in "{}();":
        i += 1
    raw = text[start:i]
    if not raw:
        return None, max(i + 1, index + 1)
    if raw.startswith("$"):
        return FoamKey(raw, raw, FoamTokenKind.EXPANSION, PatternState.INDETERMINATE), i
    if raw.startswith("#"):
        return FoamKey(raw, raw, FoamTokenKind.DIRECTIVE, PatternState.INDETERMINATE), i
    # OpenFOAM bare words are treated as literals for boundary keys. Pattern
    # selectors are normally quoted strings (wordRe stream semantics).
    return FoamKey(raw, raw, FoamTokenKind.WORD, PatternState.LITERAL), i


def parse_top_level_blocks(text: str, open_index: int, closing: str) -> FoamDictionary:
    clean = strip_comments(text)
    expected_open = "(" if closing == ")" else "{"
    if open_index >= len(clean) or clean[open_index] != expected_open:
        open_index = clean.find(expected_open, max(0, open_index - 64))
        if open_index < 0:
            return FoamDictionary((), complete=False)

    entries: list[FoamEntry] = []
    i = open_index + 1
    order = 0
    complete = False
    while i < len(clean):
        i = _skip_ws(clean, i)
        if i >= len(clean):
            break
        if clean[i] == closing:
            complete = True
            break

        key, after_key = parse_key(clean, i)
        if key is None:
            i = max(after_key, i + 1)
            continue
        j = _skip_ws(clean, after_key)
        if j < len(clean) and clean[j] == "{":
            end = matching_delimiter(clean, j, "{", "}")
            if end is None:
                entries.append(FoamEntry(key=key, body=clean[j + 1 :], order=order, declared_type=""))
                return FoamDictionary(tuple(entries), complete=False)
            body = clean[j + 1 : end]
            entries.append(
                FoamEntry(
                    key=key,
                    body=body,
                    order=order,
                    declared_type=parse_declared_type(body),
                )
            )
            order += 1
            i = end + 1
            continue

        # Preserve dynamic/unknown top-level constructs as indeterminate evidence,
        # but skip ordinary scalar/list entries that are not dictionary blocks.
        if key.pattern_state == PatternState.INDETERMINATE:
            semi = _scan_to_statement_end(clean, j, closing)
            body = clean[j:semi]
            entries.append(FoamEntry(key=key, body=body, order=order, declared_type=""))
            order += 1
            i = semi + 1
        else:
            i = _scan_to_statement_end(clean, j, closing) + 1
    return FoamDictionary(tuple(entries), complete=complete)


def parse_declared_type(body: str) -> str:
    match = _TYPE_RE.search(body)
    if match is None:
        return ""
    declared = re.sub(r"\s+", " ", match.group(1)).strip()
    return declared.split()[0] if declared else ""


def parse_in_groups(body: str) -> frozenset[str]:
    match = _GROUPS_RE.search(body)
    if match is None:
        return frozenset()
    values: set[str] = set()
    chunk = match.group(1)
    i = 0
    while i < len(chunk):
        key, nxt = parse_key(chunk, i)
        if key is None:
            i = max(nxt, i + 1)
            continue
        if key.pattern_state == PatternState.LITERAL and key.value:
            values.add(key.value)
        i = max(nxt, i + 1)
    return frozenset(values)


def find_boundary_list_start(text: str) -> int | None:
    clean = strip_comments(text)
    # polyBoundaryMesh boundary file starts with patch count followed by list.
    match = re.search(r"(?:^|\n)\s*\d+\s*\n?\s*\(", clean)
    if match is None:
        return None
    return clean.find("(", match.start())


def find_named_dictionary(text: str, name: str) -> int | None:
    clean = strip_comments(text)
    match = re.search(rf"\b{re.escape(name)}\b", clean)
    if match is None:
        return None
    brace = clean.find("{", match.end())
    return brace if brace >= 0 else None


def matching_delimiter(text: str, open_index: int, opener: str, closer: str) -> int | None:
    depth = 0
    quote: str | None = None
    i = open_index
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
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _scan_to_statement_end(text: str, index: int, section_closing: str) -> int:
    depth_round = depth_square = depth_curly = 0
    quote: str | None = None
    i = index
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
        elif ch == "(": depth_round += 1
        elif ch == ")":
            if depth_round == 0 and section_closing == ")": return i
            depth_round = max(0, depth_round - 1)
        elif ch == "[": depth_square += 1
        elif ch == "]": depth_square = max(0, depth_square - 1)
        elif ch == "{": depth_curly += 1
        elif ch == "}":
            if depth_curly == 0 and section_closing == "}": return i
            depth_curly = max(0, depth_curly - 1)
        elif ch == ";" and depth_round == depth_square == depth_curly == 0:
            return i
        i += 1
    return max(index, len(text) - 1)


def parse_named_dictionary_assignments(text: str, name: str) -> tuple[dict[str, str], bool]:
    """Parse literal scalar assignments from one named OpenFOAM dictionary.

    This is intentionally a semantic projection, not a general-purpose parser. It is
    used for contracts such as controlDict.regionSolvers where OpenFOAM itself expects
    a dictionary of region -> solver words. Dynamic directives/expansions are reported
    as incomplete rather than guessed.
    """
    clean = strip_comments(text)
    brace = find_named_dictionary(clean, name)
    if brace is None:
        return {}, False
    end = matching_delimiter(clean, brace, "{", "}")
    if end is None:
        return {}, False
    body = clean[brace + 1 : end]
    result: dict[str, str] = {}
    complete = True
    i = 0
    while i < len(body):
        i = _skip_ws(body, i)
        if i >= len(body):
            break
        key, after_key = parse_key(body, i)
        if key is None:
            i = max(i + 1, after_key)
            continue
        if key.pattern_state != PatternState.LITERAL:
            complete = False
            i = _scan_to_statement_end(body, after_key, "}") + 1
            continue
        j = _skip_ws(body, after_key)
        if j < len(body) and body[j] == "{":
            complete = False
            nested_end = matching_delimiter(body, j, "{", "}")
            i = len(body) if nested_end is None else nested_end + 1
            continue
        semi = _scan_to_statement_end(body, j, "}")
        raw_value = body[j:semi].strip()
        if not raw_value or raw_value.startswith(("$", "#")):
            complete = False
        else:
            value = raw_value.split()[0].strip('"\'')
            if value:
                result[key.value] = value
            else:
                complete = False
        i = semi + 1
    return result, complete
