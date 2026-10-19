"""Extract the final JSON answer from a free-text (chain-of-thought) response."""
import json
import re

FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)


def strip_fences(text):
    """Replace ```json ... ``` blocks by their content."""
    return FENCE.sub(lambda m: "\n" + m.group(1) + "\n", text)


def find_top_level_objects(text):
    """All balanced top-level {...} spans, ignoring braces inside JSON strings."""
    spans, depth, start = [], 0, None
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if depth > 0 and ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    return spans


def parse_response(text):
    """Return dict(parsed, parse_ok, n_objects, chose_last).

    parsed is the LAST top-level object that json-parses and has a list `slos`.
    chose_last is False when a later object existed but was not usable.
    """
    objs = find_top_level_objects(strip_fences(text or ""))
    chosen, chosen_i = None, None
    for i, s in enumerate(objs):
        try:
            o = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and isinstance(o.get("slos"), list):
            chosen, chosen_i = o, i
    return {
        "parsed": chosen,
        "parse_ok": chosen is not None,
        "n_objects": len(objs),
        "chose_last": chosen is not None and chosen_i == len(objs) - 1,
    }
