"""Few-shot prompt construction for the Intent-to-SLO experiments (design only; nothing here calls a model).

Two prompts are built from six curated examples (M1-M6) plus the paper's three visible examples (P1-P3):

  fs1_verbatim+3_prompt.txt   paper system prompt + paper examples P1-P3 verbatim + M4, M5, M6
  fs2_unitFixed+6_prompt.txt  unit-fixed system prompt with a snapshot-verification rule + M1-M6

Every example output is COMPUTED from its example snapshot (layer filter, null skipping, min/max, unit
conversion), never typed by hand, so the demonstrated labels follow the gold-label conventions exactly.
"""
import json
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PAPER_SYSTEM_PATH = PROMPT_DIR / "zs_cot_p_paper_system.txt"
FS1_NAME = "fs1_verbatim+3_prompt.txt"
FS2_NAME = "fs2_unitFixed+6_prompt.txt"
SYSTEM_MARK, USER_MARK = "=== SYSTEM ===", "=== USER ==="

# metric -> (SLO type, canonical unit, snapshot path) ; paths follow the real snapshot
METRICS = {
    "cpu_capacity": ("static_compute_resource", "millicores", ("cpu_capacity_millicores",)),
    "memory_capacity": ("static_compute_resource", "bytes", ("memory_capacity_bytes",)),
    "storage_capacity": ("static_compute_resource", "bytes", ("utilization_now", "storage_total_bytes")),
    "cpu_utilization": ("dynamic_compute_resource", "percent", ("utilization_now", "cpu_util_percent")),
    "memory_utilization": ("dynamic_compute_resource", "percent", ("utilization_now", "memory_util_percent")),
    "storage_utilization": ("dynamic_compute_resource", "percent", ("utilization_now", "storage_util_percent")),
    "port_utilization": ("node_network_resource", "percent", ("utilization_now", "port_util_percent")),
}
KB = 1024  # binary data sizes: this is the dataset's convention (1 GB = 2**30 bytes)


# ---------------------------------------------------------------- snapshot + SLO builders
def node(name, layer, cpu_m, mem_b, cpu=None, mem=None, storage=None, storage_total=None, port=None):
    """A node in the real snapshot schema, trimmed to the grounding fields. All util args None = a node with
    null utilization (like `edge-node` in the real snapshot)."""
    return {"name": name, "layer": layer, "cpu_capacity_millicores": cpu_m, "memory_capacity_bytes": mem_b,
            "utilization_now": {"cpu_util_percent": cpu, "memory_util_percent": mem,
                                "storage_total_bytes": storage_total, "storage_util_percent": storage,
                                "port_util_percent": port}}


def _value(nd, metric):
    v = nd
    for k in METRICS[metric][2]:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def slo(metric, cmp_, op, value):
    typ, unit, _ = METRICS[metric]
    return {"metric": metric, "target": {"cmp": cmp_, "op": op, "value": value}, "type": typ, "unit": unit}


def placement(layer):
    return {"metric": None, "target": {"cmp": None, "op": "set", "value": layer}, "type": "placement_location", "unit": None}


def explicit(metric, cmp_, value):
    return slo(metric, cmp_, None, value)


def implicit(snapshot, metric, op, layer=None):
    """highest/lowest over the nodes of `layer` (all nodes if None), skipping nulls; cmp >= for highest, <= for lowest."""
    vals = [_value(n, metric) for n in snapshot["nodes"] if layer in (None, n["layer"]) and _value(n, metric) is not None]
    return slo(metric, ">=" if op == "highest" else "<=", op, (max if op == "highest" else min)(vals))


# ---------------------------------------------------------------- the six curated examples
def _examples():
    ex = {}

    # M1  EEP  placement + CPU cores -> millicores + memory GB -> binary bytes (units)
    s = {"nodes": [node("cloud-a", "cloud", 2000.0, 4125683712, 6.8, 41.2, 33.0, 41133031424, 0.052),
                   node("edge-a", "edge", 4000.0, 8181379072, 9.1, 47.9, 28.0, 62220779520, 0.141)]}
    ex["M1"] = dict(service="svc/example-billing", snapshot=s,
        expression="Please get svc/example-billing running in the cloud; the target node must have at least 4 CPU cores and at least 8 GB of memory.",
        slos=[placement("cloud"), explicit("cpu_capacity", ">=", 4000), explicit("memory_capacity", ">=", 8 * KB ** 3)])

    # M2  E  single explicit storage capacity in GB (binary), no placement, snapshot present but not needed
    s = {"nodes": [node("edge-a", "edge", 4000.0, 8181379072, 11.3, 52.0, 37.0, 31914983424, 0.083),
                   node("cloud-a", "cloud", 2000.0, 2003343360, 4.4, 38.6, 61.0, 41133031424, 0.049)]}
    ex["M2"] = dict(service="svc/example-search", snapshot=s,
        expression="For svc/example-search, the target node needs at least 100 GB of storage capacity.",
        slos=[explicit("storage_capacity", ">=", 100 * KB ** 3)])

    # M3  EP  placement + explicit utilisation threshold (op null, cmp <=)
    s = {"nodes": [node("edge-a", "edge", 4000.0, 8181379072, 38.5, 44.0, 25.0, 62220779520, 0.092),
                   node("cloud-a", "cloud", 2000.0, 4125683712, 12.7, 40.3, 52.0, 41133031424, 0.066)]}
    ex["M3"] = dict(service="svc/example-gateway", snapshot=s,
        expression="Can you put svc/example-gateway on edge nodes and keep CPU utilization below 45%?",
        slos=[placement("edge"), explicit("cpu_utilization", "<=", 45)])

    # M4  EIP  placement + threshold + LOWEST cpu utilisation restricted to the edge layer (a cloud node is lower
    #          globally = the trap; one edge node has null utilisation and must be skipped)
    s = {"nodes": [node("edge-a", "edge", 4000.0, 8181379072, 9.8, 46.1, 23.0, 62220779520, 0.168),
                   node("edge-b", "edge", 4000.0, 3968221184, 4.1, 34.3, 18.0, 31914983424, 0.049),
                   node("edge-c", "edge", 4000.0, 8318894080),
                   node("cloud-a", "cloud", 2000.0, 2003343360, 1.2, 36.5, 55.0, 8132173824, 0.045)]}
    ex["M4"] = dict(service="svc/example-worker", snapshot=s,
        expression="I would like svc/example-worker on the edge layer, with storage utilization below 65%, on the node with the lowest current CPU utilization.",
        slos=[placement("edge"), explicit("storage_utilization", "<=", 65), implicit(s, "cpu_utilization", "lowest", "edge")])

    # M5  IIP  cloud layer: HIGHEST memory capacity + LOWEST port utilisation (an edge node is larger / lower = trap)
    s = {"nodes": [node("cloud-a", "cloud", 2000.0, 2003343360, 3.3, 36.9, 57.0, 8132173824, 0.047),
                   node("cloud-b", "cloud", 2000.0, 4125683712, 5.6, 40.2, 49.0, 8132173824, 0.052),
                   node("edge-a", "edge", 4000.0, 8181379072, 9.8, 46.6, 23.0, 62220779520, 0.011)]}
    ex["M5"] = dict(service="svc/example-queue", snapshot=s,
        expression="Schedule svc/example-queue on the cloud side, on the node with the highest memory capacity and the lowest port utilization.",
        slos=[placement("cloud"), implicit(s, "memory_capacity", "highest", "cloud"), implicit(s, "port_utilization", "lowest", "cloud")])

    # M6  EIIP  L4-style: placement + threshold + LOWEST storage utilisation + HIGHEST storage capacity on the edge layer
    #          (cloud node is lower globally; an edge node has null utilisation AND null storage total)
    s = {"nodes": [node("edge-a", "edge", 4000.0, 8181379072, 6.2, 48.8, 23.0, 31914983424, 0.131),
                   node("edge-b", "edge", 4000.0, 3968221184, 7.9, 33.5, 41.5, 62220779520, 0.073),
                   node("edge-c", "edge", 4000.0, 8318894080),
                   node("cloud-a", "cloud", 2000.0, 2003343360, 2.1, 36.5, 11.0, 8132173824, 0.045)]}
    ex["M6"] = dict(service="svc/example-cache", snapshot=s,
        expression="Go ahead and run svc/example-cache on the edge side, keep memory utilization below 60%, choose the node with the lowest current storage utilization, and the node with the highest storage capacity.",
        slos=[placement("edge"), explicit("memory_utilization", "<=", 60),
              implicit(s, "storage_utilization", "lowest", "edge"), implicit(s, "storage_capacity", "highest", "edge")])
    return ex


EXAMPLES = _examples()

# the paper's three visible examples, transcribed verbatim from Fig. 9 (rendered form: f-string braces resolved)
PAPER_EXAMPLES = {
    "P1": '''Input:
{
  "service": "svc/example-auth",
  "expression": "Deploy svc/example-auth, keep CPU utilization below 55 percent.",
  "snapshot": {"nodes": [{"name": "n1", "cpu_utilization": 40}, {"name": "n2", "cpu_utilization": 62}]}
}
Output:
{
  "service": "svc/example-auth",
  "slos": [
    {
      "metric": "cpu_utilization",
      "target": {"cmp": "<=", "op": null, "value": 55, "range": null},
      "type": "dynamic_compute_resource",
      "unit": "percent"
    }
  ]
}''',
    "P2": '''Input:
{
  "service": "svc/example-cache",
  "expression": "Run svc/example-cache with at least 2 GB memory.",
  "snapshot": {"nodes": [{"name": "n1", "memory_capacity": 1073741824}, {"name": "n2", "memory_capacity": 8589934592}]}
}
Output:
{
  "service": "svc/example-cache",
  "slos": [
    {
      "metric": "memory_capacity",
      "target": {"cmp": ">=", "op": null, "value": 2000000000, "range": null},
      "type": "static_compute_resource",
      "unit": "bytes"
    }
  ]
}''',
    "P3": '''Input:
{
  "service": "svc/example-worker",
  "expression": "Deploy svc/example-worker with at least 500 MiB of storage capacity.",
  "snapshot": {"nodes": [{"name": "n1", "storage_capacity": 800000000}, {"name": "n2", "storage_capacity": 600000000}]}
}
Output:
{
  "service": "svc/example-worker",
  "slos": [
    {
      "metric": "storage_capacity",
      "target": {"cmp": ">=", "op": null, "value": 524288000, "range": null},
      "type": "static_compute_resource",
      "unit": "bytes"
    }
  ]
}''',
}


# ---------------------------------------------------------------- rendering
def _j(v):
    return json.dumps(v)


def render_output(ex):
    lines = ["{", f'  "service": {_j(ex["service"])},', '  "slos": [']
    blocks = []
    for s in ex["slos"]:
        t = s["target"]
        blocks.append("\n".join([
            "    {", f'      "metric": {_j(s["metric"])},',
            f'      "target": {{"cmp": {_j(t["cmp"])}, "op": {_j(t["op"])}, "value": {_j(t["value"])}}},',
            f'      "type": {_j(s["type"])},', f'      "unit": {_j(s["unit"])}', "    }"]))
    lines.append(",\n".join(blocks))
    lines += ["  ]", "}"]
    return "\n".join(lines)


def render_example(ex):
    """Input/Output text of one curated example (same layout as the paper's Fig. 9)."""
    snap = json.dumps(ex["snapshot"], separators=(",", ":"))  # compact, like the real snapshot in the query
    return (f'Input:\n{{\n  "service": {_j(ex["service"])},\n  "expression": {_j(ex["expression"])},\n'
            f'  "snapshot": {snap}\n}}\nOutput:\n{render_output(ex)}')


FINAL_INPUT = '''Input:
{
  "service": "{service}",
  "expression": "{expression}",
  "snapshot": {snapshot}
}

Answer:'''


def user_template(blocks):
    body = "\n\n".join(f"Example {i}\n{b}" for i, b in enumerate(blocks, 1))
    return body + "\n\n" + FINAL_INPUT


# ---------------------------------------------------------------- system prompts
def system_fs1():
    """The paper's zero-shot system prompt, exactly as used for the zero-shot runs (unchanged)."""
    return PAPER_SYSTEM_PATH.read_text(encoding="utf-8")


UNIT_CONVENTIONS = '''Unit conventions:
- 1 core = 1000 millicores, so "4 CPU cores" means cpu_capacity 4000 (millicores).
- Data sizes are binary: 1 KB = 1 KiB = 1024 bytes; 1 MB = 1 MiB = 1024^2 bytes; 1 GB = 1 GiB = 1024^3 = 1073741824 bytes; 1 TB = 1 TiB = 1024^4 bytes. So "8 GB" means 8589934592 bytes.
- The snapshot already uses the canonical units (millicores, bytes, percent): copy its values unchanged.
'''

SNAPSHOT_RULE = '''Snapshot use (MANDATORY for highest/lowest requests):
- Explicit thresholds ("below 45%", "at least 4 CPU cores") do not need the snapshot; there target.op is null.
- If the expression asks for an extremum (highest, lowest, most, least, maximum, minimum), you MUST verify the answer in the snapshot: never leave target.value null and never guess it.
- Candidate nodes: only the nodes of the requested layer (edge or cloud) if the expression names one, otherwise all nodes. Ignore nodes whose value for that metric is null.
- Compare the metric values of the candidate nodes, take the maximum for "highest" or the minimum for "lowest", and copy that node's value unchanged into target.value.
- Set target.op to "highest" or "lowest", with cmp ">=" for highest and "<=" for lowest.
- Do not invent numeric values.
'''

# (old text in the paper prompt, new text, label) -- applied in order; each old text must occur exactly once
FS2_CHANGES = [
    ("cpu_capacity=cores;", "cpu_capacity=millicores;", "canonical cpu unit: cores -> millicores"),
    ("port_utilization=percent;\n\nProcedure:", "port_utilization=percent;\n\n" + UNIT_CONVENTIONS + "\nProcedure:",
     "explicit unit-conventions block (cores->millicores factor, binary data sizes, snapshot already canonical)"),
    ("using standard unit conventions (SI/IEC for data sizes; time and rate unit conversions as standard).",
     "using the unit conventions above (time and rate unit conversions as standard).",
     "removes the ambiguous 'SI/IEC' wording"),
    ("- Emit the JSON.\n", "- If the expression asks for an extremum, follow \"Snapshot use\" below before emitting.\n- Emit the JSON.\n",
     "procedure step that points to the snapshot rule"),
    ('''Snapshot use:
- Only consult the snapshot when the expression requests an extremum (highest/lowest/etc.).
- When comparing snapshot values, normalize them to canonical units first, then choose max/min.
- Do not invent numeric values.
''', SNAPSHOT_RULE, "precautionary snapshot-verification rule (replaces the paper's 'Snapshot use' block)"),
]


def system_fs2():
    s = system_fs1()
    for old, new, _ in FS2_CHANGES:
        assert s.count(old) == 1, f"change target not found exactly once: {old[:50]!r}"
        s = s.replace(old, new)
    return s


# ---------------------------------------------------------------- the two prompts
def fs1_blocks():
    return [PAPER_EXAMPLES["P1"], PAPER_EXAMPLES["P2"], PAPER_EXAMPLES["P3"]] + [render_example(EXAMPLES[k]) for k in ("M4", "M5", "M6")]


def fs2_blocks():
    return [render_example(EXAMPLES[k]) for k in ("M1", "M2", "M3", "M4", "M5", "M6")]


def build(name):
    """(system, user_template) of a prompt; user_template keeps the {service} {expression} {snapshot} placeholders."""
    if name == FS1_NAME:
        return system_fs1(), user_template(fs1_blocks())
    if name == FS2_NAME:
        return system_fs2(), user_template(fs2_blocks())
    raise KeyError(name)


def file_text(name):
    system, user = build(name)
    return f"{SYSTEM_MARK}\n{system.rstrip()}\n{USER_MARK}\n{user}\n"


def write_prompt_files(directory=PROMPT_DIR):
    for name in (FS1_NAME, FS2_NAME):
        (Path(directory) / name).write_text(file_text(name), encoding="utf-8")


if __name__ == "__main__":
    write_prompt_files()
    print("wrote", FS1_NAME, "and", FS2_NAME, "to", PROMPT_DIR)
