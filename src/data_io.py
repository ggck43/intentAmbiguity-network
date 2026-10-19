"""Read-only loaders for the Intent Engine dataset and infrastructure snapshot."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET_PATH = RAW / "dataset.json"
SNAPSHOT_PATH = RAW / "snapshot_2026-04-25T23-12-00.393306Z.json"

# metric -> path into a snapshot node (spec section 2)
SNAPSHOT_FIELDS = {
    "cpu_capacity": ("cpu_capacity_millicores",),
    "memory_capacity": ("memory_capacity_bytes",),
    "storage_capacity": ("utilization_now", "storage_total_bytes"),
    "cpu_utilization": ("utilization_now", "cpu_util_percent"),
    "memory_utilization": ("utilization_now", "memory_util_percent"),
    "storage_utilization": ("utilization_now", "storage_util_percent"),
    "port_utilization": ("utilization_now", "port_util_percent"),
}


def load_dataset(path=DATASET_PATH):
    with open(path) as f:
        return json.load(f)


def load_records(path=DATASET_PATH):
    return load_dataset(path)["data"]


def load_snapshot(path=SNAPSHOT_PATH):
    with open(path) as f:
        return json.load(f)


def snapshot_compact(snap):
    """Snapshot string as it goes into the user prompt."""
    return json.dumps(snap, separators=(",", ":"))


def node_value(node, metric):
    v = node
    for key in SNAPSHOT_FIELDS[metric]:
        v = v.get(key) if isinstance(v, dict) else None
    return v


def metric_table(snap):
    """Rows of (node, layer, {metric: value}) for every node."""
    return [(n["name"], n["layer"], {m: node_value(n, m) for m in SNAPSHOT_FIELDS})
            for n in snap["nodes"]]


def template_id(record):
    intent = record["input"]["intent"]
    return intent["expression"].replace(intent["service"], "<SVC>").rstrip(". ")
