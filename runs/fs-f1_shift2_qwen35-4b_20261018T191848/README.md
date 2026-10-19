# Few-shot fs1_verbatim+3_prompt.txt (fs-f1), Qwen/Qwen3.5-4B

- date: 2026-10-18T19:18:48
- git commit: None
- model revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
- subset: valid (521 records)
- expression file: data/shift/shift2_valid.json
- thinking: False
- prompt variant: fs1_verbatim+3_prompt.txt
- max_num_seqs: vLLM default, gpu_memory_utilization: 0.7
- decoding: temperature=0, top_p=1, max_tokens=2048, seed=0
- vLLM: 0.31.0, batch=64 records per call, max_model_len=12288 (greedy outputs are reproducible for the same batch size, not across batch sizes)
- snapshot: snapshot_2026-04-25T23-12-00.393306Z.json (5023 tokens)
- system prompt sha256: 6439b74e6bacb8f983e380ecd8b5750a16c8697108a8d44e0beda749691a3e4c
- user template sha256: f05f53ffffb31c4d94b7f1ab1f201a59cc3c53ab54d619abd8dbe108f53ea553
- dataset sha256[:12]: f5a4fe704b5b

## Deviations / confounds

1-2. Few-shot prompt file (design and justification: notebook 3.fs_promptDesign.ipynb). The examples sit in the single
   user message (paper format); the paper's three system-prompt header lines (Service / Intent Expression / Snapshot
   placeholders) are omitted, as in every zero-shot run. fs1 keeps the paper's system prompt (cores) and its decimal-GB
   example; fs2 states millicores and binary data sizes and has a mandatory snapshot-verification rule.
4. Qwen's native thinking mode is disabled (chat template enable_thinking=False) so the only reasoning is the
   prompted CoT.
3. Backend is a local vLLM model, not an API: no system_fingerprint (logged as null); latency_s is the
   batch wall time divided by batch size, not a per-request latency.
5. Strict S0 scoring: no unit conversion or tolerance. Gold stores cpu_capacity in millicores while the paper's
   prompt asks for cores, so correct cpu_capacity answers count as FP+FN. Gold highest/lowest values are taken
   layer-filtered in the real records but globally in the synthetic ones.

## System prompt (fs1_verbatim+3_prompt.txt)

```
Instruction:
You are an information extraction system that converts natural-language service placement intents into SLO labels.

IMPORTANT:
Output MUST be JSON only.

Return ONLY valid JSON matching:
{
 "service": "<service>",
 "slos": [
  {
   "metric": <string or null>,
   "target": {
    "cmp": "<=" | ">=" | "between" | "=" | null,
    "op": "highest" | "lowest" | "set" | null,
    "value": <number | string | null>
   },
   "type": "placement_location" | "static_compute_resource" | "dynamic_compute_resource" | "node_network_resource",
   "unit": <string or null>
  }
 ]
}

Constraints:
1) Use only these types/metrics:
- static_compute_resource: cpu_capacity, memory_capacity, storage_capacity
- dynamic_compute_resource: cpu_utilization, memory_utilization, storage_utilization
- node_network_resource: bandwidth, port_utilization
- placement_location: metric=null, unit=null

2) Use canonical units for output:
cpu_capacity=cores;
memory_capacity=bytes;
storage_capacity=bytes;
*_utilization=percent;
bandwidth=Mbps;
port_utilization=percent;

Procedure:
- Identify each constraint in the expression.
- Parse numeric values and their units (if any).
- Normalize numeric values into the canonical unit for that metric using standard unit conventions (SI/IEC for data sizes; time and rate unit conversions as standard).
- Emit the JSON.

Snapshot use:
- Only consult the snapshot when the expression requests an extremum (highest/lowest/etc.).
- When comparing snapshot values, normalize them to canonical units first, then choose max/min.
- Do not invent numeric values.

Placement:
- Only emit placement_location if edge/cloud/both/any is explicitly stated;
  encode with target.op="set" and target.value.
```

## User message template (fs1_verbatim+3_prompt.txt)

```
Example 1
Input:
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
}

Example 2
Input:
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
}

Example 3
Input:
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
}

Example 4
Input:
{
  "service": "svc/example-worker",
  "expression": "I would like svc/example-worker on the edge layer, with storage utilization below 65%, on the node with the lowest current CPU utilization.",
  "snapshot": {"nodes":[{"name":"edge-a","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8181379072,"utilization_now":{"cpu_util_percent":9.8,"memory_util_percent":46.1,"storage_total_bytes":62220779520,"storage_util_percent":23.0,"port_util_percent":0.168}},{"name":"edge-b","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":3968221184,"utilization_now":{"cpu_util_percent":4.1,"memory_util_percent":34.3,"storage_total_bytes":31914983424,"storage_util_percent":18.0,"port_util_percent":0.049}},{"name":"edge-c","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8318894080,"utilization_now":{"cpu_util_percent":null,"memory_util_percent":null,"storage_total_bytes":null,"storage_util_percent":null,"port_util_percent":null}},{"name":"cloud-a","layer":"cloud","cpu_capacity_millicores":2000.0,"memory_capacity_bytes":2003343360,"utilization_now":{"cpu_util_percent":1.2,"memory_util_percent":36.5,"storage_total_bytes":8132173824,"storage_util_percent":55.0,"port_util_percent":0.045}}]}
}
Output:
{
  "service": "svc/example-worker",
  "slos": [
    {
      "metric": null,
      "target": {"cmp": null, "op": "set", "value": "edge"},
      "type": "placement_location",
      "unit": null
    },
    {
      "metric": "storage_utilization",
      "target": {"cmp": "<=", "op": null, "value": 65},
      "type": "dynamic_compute_resource",
      "unit": "percent"
    },
    {
      "metric": "cpu_utilization",
      "target": {"cmp": "<=", "op": "lowest", "value": 4.1},
      "type": "dynamic_compute_resource",
      "unit": "percent"
    }
  ]
}

Example 5
Input:
{
  "service": "svc/example-queue",
  "expression": "Schedule svc/example-queue on the cloud side, on the node with the highest memory capacity and the lowest port utilization.",
  "snapshot": {"nodes":[{"name":"cloud-a","layer":"cloud","cpu_capacity_millicores":2000.0,"memory_capacity_bytes":2003343360,"utilization_now":{"cpu_util_percent":3.3,"memory_util_percent":36.9,"storage_total_bytes":8132173824,"storage_util_percent":57.0,"port_util_percent":0.047}},{"name":"cloud-b","layer":"cloud","cpu_capacity_millicores":2000.0,"memory_capacity_bytes":4125683712,"utilization_now":{"cpu_util_percent":5.6,"memory_util_percent":40.2,"storage_total_bytes":8132173824,"storage_util_percent":49.0,"port_util_percent":0.052}},{"name":"edge-a","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8181379072,"utilization_now":{"cpu_util_percent":9.8,"memory_util_percent":46.6,"storage_total_bytes":62220779520,"storage_util_percent":23.0,"port_util_percent":0.011}}]}
}
Output:
{
  "service": "svc/example-queue",
  "slos": [
    {
      "metric": null,
      "target": {"cmp": null, "op": "set", "value": "cloud"},
      "type": "placement_location",
      "unit": null
    },
    {
      "metric": "memory_capacity",
      "target": {"cmp": ">=", "op": "highest", "value": 4125683712},
      "type": "static_compute_resource",
      "unit": "bytes"
    },
    {
      "metric": "port_utilization",
      "target": {"cmp": "<=", "op": "lowest", "value": 0.047},
      "type": "node_network_resource",
      "unit": "percent"
    }
  ]
}

Example 6
Input:
{
  "service": "svc/example-cache",
  "expression": "Go ahead and run svc/example-cache on the edge side, keep memory utilization below 60%, choose the node with the lowest current storage utilization, and the node with the highest storage capacity.",
  "snapshot": {"nodes":[{"name":"edge-a","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8181379072,"utilization_now":{"cpu_util_percent":6.2,"memory_util_percent":48.8,"storage_total_bytes":31914983424,"storage_util_percent":23.0,"port_util_percent":0.131}},{"name":"edge-b","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":3968221184,"utilization_now":{"cpu_util_percent":7.9,"memory_util_percent":33.5,"storage_total_bytes":62220779520,"storage_util_percent":41.5,"port_util_percent":0.073}},{"name":"edge-c","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8318894080,"utilization_now":{"cpu_util_percent":null,"memory_util_percent":null,"storage_total_bytes":null,"storage_util_percent":null,"port_util_percent":null}},{"name":"cloud-a","layer":"cloud","cpu_capacity_millicores":2000.0,"memory_capacity_bytes":2003343360,"utilization_now":{"cpu_util_percent":2.1,"memory_util_percent":36.5,"storage_total_bytes":8132173824,"storage_util_percent":11.0,"port_util_percent":0.045}}]}
}
Output:
{
  "service": "svc/example-cache",
  "slos": [
    {
      "metric": null,
      "target": {"cmp": null, "op": "set", "value": "edge"},
      "type": "placement_location",
      "unit": null
    },
    {
      "metric": "memory_utilization",
      "target": {"cmp": "<=", "op": null, "value": 60},
      "type": "dynamic_compute_resource",
      "unit": "percent"
    },
    {
      "metric": "storage_utilization",
      "target": {"cmp": "<=", "op": "lowest", "value": 23.0},
      "type": "dynamic_compute_resource",
      "unit": "percent"
    },
    {
      "metric": "storage_capacity",
      "target": {"cmp": ">=", "op": "highest", "value": 62220779520},
      "type": "static_compute_resource",
      "unit": "bytes"
    }
  ]
}

Input:
{
  "service": "{service}",
  "expression": "{expression}",
  "snapshot": {snapshot}
}

Answer:
```
