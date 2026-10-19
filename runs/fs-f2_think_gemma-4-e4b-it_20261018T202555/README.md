# Few-shot fs2_unitFixed+6_prompt.txt + native thinking (fs-f2_think), google/gemma-4-E4B-it

- date: 2026-10-18T20:25:55
- git commit: None
- model revision: ee0ef6023621cff504d758262d4e04895a5af4a2
- subset: valid (521 records)
- expression file: none (original expressions)
- thinking: True
- prompt variant: fs2_unitFixed+6_prompt.txt
- max_num_seqs: vLLM default, gpu_memory_utilization: 0.7
- decoding: temperature=0, top_p=1, think budget=4096, answer max_tokens=1024, seed=0
- vLLM: 0.31.0, batch=64 records per call, max_model_len=16384 (greedy outputs are reproducible for the same batch size, not across batch sizes)
- snapshot: snapshot_2026-04-25T23-12-00.393306Z.json (5455 tokens)
- system prompt sha256: 08ed6cb27bc22e14516628366eef90fb3c021080757e500e8fc19acd3b1aabe9
- user template sha256: c3088de53179924d703159020e1d959403241b15c0304277bc4007d54d511637
- dataset sha256[:12]: f5a4fe704b5b

## Deviations / confounds

1-2. Few-shot prompt file (design and justification: notebook 3.fs_promptDesign.ipynb). The examples sit in the single
   user message (paper format); the paper's three system-prompt header lines (Service / Intent Expression / Snapshot
   placeholders) are omitted, as in every zero-shot run. fs1 keeps the paper's system prompt (cores) and its decimal-GB
   example; fs2 states millicores and binary data sizes and has a mandatory snapshot-verification rule.
7. Gemma 4 (instruction-tuned) is run through its own chat template with a normal system turn. Thinking is switched with
   enable_thinking (the template puts <|think|> into the system turn). Greedy decoding is used for comparability with the
   Qwen runs although the model card recommends temperature 1.0, top_p 0.95, top_k 64. Thinking is budgeted in two stages
   as for Qwen: stage 1 stops at <channel|> (token id of the end-of-thought marker) or after --think-budget tokens; if the
   thought was not closed the standard budget-forcing sentence plus <channel|> is appended; stage 2 generates the answer.
   Only the stage-2 text is parsed and scored. Without thinking the model answers directly.
3. Backend is a local vLLM model, not an API: no system_fingerprint (logged as null); latency_s is the
   batch wall time divided by batch size, not a per-request latency.
5. Strict S0 scoring: no unit conversion or tolerance. Gold stores cpu_capacity in millicores while the paper's
   prompt asks for cores, so correct cpu_capacity answers count as FP+FN. Gold highest/lowest values are taken
   layer-filtered in the real records but globally in the synthetic ones.

## System prompt (fs2_unitFixed+6_prompt.txt)

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
cpu_capacity=millicores;
memory_capacity=bytes;
storage_capacity=bytes;
*_utilization=percent;
bandwidth=Mbps;
port_utilization=percent;

Unit conventions:
- 1 core = 1000 millicores, so "4 CPU cores" means cpu_capacity 4000 (millicores).
- Data sizes are binary: 1 KB = 1 KiB = 1024 bytes; 1 MB = 1 MiB = 1024^2 bytes; 1 GB = 1 GiB = 1024^3 = 1073741824 bytes; 1 TB = 1 TiB = 1024^4 bytes. So "8 GB" means 8589934592 bytes.
- The snapshot already uses the canonical units (millicores, bytes, percent): copy its values unchanged.

Procedure:
- Identify each constraint in the expression.
- Parse numeric values and their units (if any).
- Normalize numeric values into the canonical unit for that metric using the unit conventions above (time and rate unit conversions as standard).
- If the expression asks for an extremum, follow "Snapshot use" below before emitting.
- Emit the JSON.

Snapshot use (MANDATORY for highest/lowest requests):
- Explicit thresholds ("below 45%", "at least 4 CPU cores") do not need the snapshot; there target.op is null.
- If the expression asks for an extremum (highest, lowest, most, least, maximum, minimum), you MUST verify the answer in the snapshot: never leave target.value null and never guess it.
- Candidate nodes: only the nodes of the requested layer (edge or cloud) if the expression names one, otherwise all nodes. Ignore nodes whose value for that metric is null.
- Compare the metric values of the candidate nodes, take the maximum for "highest" or the minimum for "lowest", and copy that node's value unchanged into target.value.
- Set target.op to "highest" or "lowest", with cmp ">=" for highest and "<=" for lowest.
- Do not invent numeric values.

Placement:
- Only emit placement_location if edge/cloud/both/any is explicitly stated;
  encode with target.op="set" and target.value.
```

## User message template (fs2_unitFixed+6_prompt.txt)

```
Example 1
Input:
{
  "service": "svc/example-billing",
  "expression": "Please get svc/example-billing running in the cloud; the target node must have at least 4 CPU cores and at least 8 GB of memory.",
  "snapshot": {"nodes":[{"name":"cloud-a","layer":"cloud","cpu_capacity_millicores":2000.0,"memory_capacity_bytes":4125683712,"utilization_now":{"cpu_util_percent":6.8,"memory_util_percent":41.2,"storage_total_bytes":41133031424,"storage_util_percent":33.0,"port_util_percent":0.052}},{"name":"edge-a","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8181379072,"utilization_now":{"cpu_util_percent":9.1,"memory_util_percent":47.9,"storage_total_bytes":62220779520,"storage_util_percent":28.0,"port_util_percent":0.141}}]}
}
Output:
{
  "service": "svc/example-billing",
  "slos": [
    {
      "metric": null,
      "target": {"cmp": null, "op": "set", "value": "cloud"},
      "type": "placement_location",
      "unit": null
    },
    {
      "metric": "cpu_capacity",
      "target": {"cmp": ">=", "op": null, "value": 4000},
      "type": "static_compute_resource",
      "unit": "millicores"
    },
    {
      "metric": "memory_capacity",
      "target": {"cmp": ">=", "op": null, "value": 8589934592},
      "type": "static_compute_resource",
      "unit": "bytes"
    }
  ]
}

Example 2
Input:
{
  "service": "svc/example-search",
  "expression": "For svc/example-search, the target node needs at least 100 GB of storage capacity.",
  "snapshot": {"nodes":[{"name":"edge-a","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8181379072,"utilization_now":{"cpu_util_percent":11.3,"memory_util_percent":52.0,"storage_total_bytes":31914983424,"storage_util_percent":37.0,"port_util_percent":0.083}},{"name":"cloud-a","layer":"cloud","cpu_capacity_millicores":2000.0,"memory_capacity_bytes":2003343360,"utilization_now":{"cpu_util_percent":4.4,"memory_util_percent":38.6,"storage_total_bytes":41133031424,"storage_util_percent":61.0,"port_util_percent":0.049}}]}
}
Output:
{
  "service": "svc/example-search",
  "slos": [
    {
      "metric": "storage_capacity",
      "target": {"cmp": ">=", "op": null, "value": 107374182400},
      "type": "static_compute_resource",
      "unit": "bytes"
    }
  ]
}

Example 3
Input:
{
  "service": "svc/example-gateway",
  "expression": "Can you put svc/example-gateway on edge nodes and keep CPU utilization below 45%?",
  "snapshot": {"nodes":[{"name":"edge-a","layer":"edge","cpu_capacity_millicores":4000.0,"memory_capacity_bytes":8181379072,"utilization_now":{"cpu_util_percent":38.5,"memory_util_percent":44.0,"storage_total_bytes":62220779520,"storage_util_percent":25.0,"port_util_percent":0.092}},{"name":"cloud-a","layer":"cloud","cpu_capacity_millicores":2000.0,"memory_capacity_bytes":4125683712,"utilization_now":{"cpu_util_percent":12.7,"memory_util_percent":40.3,"storage_total_bytes":41133031424,"storage_util_percent":52.0,"port_util_percent":0.066}}]}
}
Output:
{
  "service": "svc/example-gateway",
  "slos": [
    {
      "metric": null,
      "target": {"cmp": null, "op": "set", "value": "edge"},
      "type": "placement_location",
      "unit": null
    },
    {
      "metric": "cpu_utilization",
      "target": {"cmp": "<=", "op": null, "value": 45},
      "type": "dynamic_compute_resource",
      "unit": "percent"
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
