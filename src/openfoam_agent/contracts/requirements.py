"""Chronological explicit assignments, with conservative supersession evidence."""
from __future__ import annotations
import re
import unicodedata

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ASSIGN = re.compile(r"(?P<key>[\w.]+)\s*(?:=|:)\s*(?P<value>"+_NUMBER+r")", re.UNICODE)
# This does not infer physical targets. Aliases only normalize explicit names.
_ALIASES = {"re": "operating.reynolds_number", "reynolds": "operating.reynolds_number",
            "reynolds_number": "operating.reynolds_number", "\ub808\uc774\ub180\uc988\uc218": "operating.reynolds_number"}
_QUESTION = re.compile(r"\?|\b(?:why|would|what if)\b|\uc65c|\uc5b4\ub54c|\uc77c\uae4c|\uc778\uac00", re.I)
_CANCEL = re.compile(r"(?:cancel|remove|delete|\ucde8\uc18c|\uc0ad\uc81c)\s*[:=]?\s*([\w.]+)", re.I)


def canonical_target(key):
    normalized = unicodedata.normalize("NFKC", key).casefold()
    return _ALIASES.get(normalized, normalized)


def active_requirement_texts(turns):
    """Mask ONLY earlier values replaced/cancelled by later explicit same-key input.

    Questions, pronouns and inferred target changes are not treated as consent.
    Unnamed/free-form quantities remain subject to normal preservation checks.
    """
    occurrences = {}
    masks = [[] for _ in turns]
    history = []
    for index, text in enumerate(turns):
        is_question = bool(_QUESTION.search(text))
        for match in _ASSIGN.finditer(text):
            if is_question:
                continue
            raw_key = match.group("key")
            key = canonical_target(raw_key)
            if key not in _ALIASES.values() and "." not in raw_key:
                # "inlet pressure" and "outlet pressure" must never collapse to pressure.
                continue
            prior = occurrences.get(key)
            if prior is not None and prior["turn"] != index:
                masks[prior["turn"]].append(prior["span"])
                history.append({"target": key, "status": "superseded", "previous_turn": prior["turn"],
                    "replacement_turn": index, "previous_evidence": prior["evidence"],
                    "replacement_evidence": match.group(0)})
            occurrences[key] = {"turn": index, "span": match.span("value"), "evidence": text[match.start():match.end()+12].split("\n")[0],
                                "value": match.group("value")}
        if not is_question:
            for match in _CANCEL.finditer(text):
                key = canonical_target(match.group(1))
                prior = occurrences.pop(key, None)
                if prior is not None:
                    masks[prior["turn"]].append(prior["span"])
                    history.append({"target": key, "status": "cancelled", "previous_turn": prior["turn"],
                        "replacement_turn": index, "previous_evidence": prior["evidence"],
                        "replacement_evidence": match.group(0)})
    active = []
    for text, ranges in zip(turns, masks):
        chars = list(text)
        for start, end in ranges:
            chars[start:end] = " " * (end-start)
        active.append("".join(chars))
    return active, history, occurrences
