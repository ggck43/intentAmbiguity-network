"""Tools for studying the controlled language of the Intent Engine dataset (analysis only, nothing here calls a model).

 - tokenisation / template normalisation used to measure vocabulary and template diversity
 - `parse_expression`: a deterministic rule-based parser written from the dataset's own clause inventory. It is a
   deliberately small set of regular expressions (see RULES); how much of the dataset it solves measures how closed
   the dataset's language is.
"""
import re

from src.fewshot import KB, METRICS

WORD = re.compile(r"[A-Za-z']+|%|\d+(?:\.\d+)?")


def tokens(expression, service="", numbers=False):
    """lower-cased tokens of an intent, with the service name removed and (by default) numbers dropped"""
    toks = WORD.findall(expression.replace(service, " ").lower()) if service else WORD.findall(expression.lower())
    return toks if numbers else [t for t in toks if not t[0].isdigit()]


def template(expression, service):
    """service -> <SVC>, numbers -> N"""
    return re.sub(r"\d+(\.\d+)?", "N", expression.replace(service, "<SVC>")).strip()


# ---------------------------------------------------------------- the deterministic parser
UTIL = {"cpu": "cpu_utilization", "memory": "memory_utilization", "storage": "storage_utilization"}
# (name, regular expression); every rule matches a clause shape that occurs in the dataset
RULES = [
    ("placement", re.compile(r"\b(edge|cloud)\b")),
    ("threshold", re.compile(r"\b(cpu|memory|storage) utilization below (\d+(?:\.\d+)?)\s*%")),
    ("cpu cores", re.compile(r"at least (\d+(?:\.\d+)?) cpu cores")),
    ("memory GB", re.compile(r"at least (\d+(?:\.\d+)?) gb of memory")),
    ("storage GB", re.compile(r"at least (\d+(?:\.\d+)?) gb of storage capacity")),
    ("lookup", re.compile(r"the node with the (highest|lowest) (?:current )?(cpu|memory|storage|port) (utilization|capacity)")),
]


def _slo(metric, cmp_, op, value):
    typ, unit, _ = METRICS[metric]
    return {"metric": metric, "target": {"cmp": cmp_, "op": op, "value": value}, "type": typ, "unit": unit}


def _lookup_value(snapshot, metric, op, layer):
    from src.fewshot import _value
    vals = [_value(n, metric) for n in snapshot["nodes"] if layer in (None, n["layer"]) and _value(n, metric) is not None]
    return (max if op == "highest" else min)(vals)


def parse_expression(expression, service, snapshot, scope="layer"):
    """Rule-based extraction of the SLO list. scope: 'layer' (extremum over the requested layer, else all nodes) or 'global'."""
    text = expression.replace(service, " ").lower()
    out = []
    m = RULES[0][1].search(text)
    layer = m.group(1) if m else None
    if layer:
        out.append({"metric": None, "target": {"cmp": None, "op": "set", "value": layer}, "type": "placement_location", "unit": None})
    # explicit clauses and lookups in order of appearance
    found = []
    for name, rx in RULES[1:]:
        for mm in rx.finditer(text):
            found.append((mm.start(), name, mm))
    for _, name, mm in sorted(found, key=lambda x: x[0]):
        if name == "threshold":
            v = float(mm.group(2)); out.append(_slo(UTIL[mm.group(1)], "<=", None, int(v) if v.is_integer() else v))
        elif name == "cpu cores":
            out.append(_slo("cpu_capacity", ">=", None, float(mm.group(1)) * 1000))
        elif name == "memory GB":
            out.append(_slo("memory_capacity", ">=", None, float(mm.group(1)) * KB ** 3))
        elif name == "storage GB":
            out.append(_slo("storage_capacity", ">=", None, float(mm.group(1)) * KB ** 3))
        elif name == "lookup":
            op, res, kind = mm.group(1), mm.group(2), mm.group(3)
            metric = f"{res}_{kind}"
            if metric in METRICS:
                out.append(_slo(metric, ">=" if op == "highest" else "<=", op, _lookup_value(snapshot, metric, op, layer if scope == "layer" else None)))
    return out
