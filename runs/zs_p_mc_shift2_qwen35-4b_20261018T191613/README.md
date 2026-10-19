# Zero-shot (zs), Qwen/Qwen3.5-4B

- date: 2026-10-18T19:16:13
- git commit: None
- model revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
- subset: valid (521 records)
- expression file: data/shift/shift2_valid.json
- thinking: False
- prompt variant: p_mc
- max_num_seqs: vLLM default, gpu_memory_utilization: 0.7
- decoding: temperature=0, top_p=1, max_tokens=2048, seed=0
- vLLM: 0.31.0, batch=64 records per call, max_model_len=8192 (greedy outputs are reproducible for the same batch size, not across batch sizes)
- snapshot: snapshot_2026-04-25T23-12-00.393306Z.json (5023 tokens)
- system prompt sha256: 3a2b4a9a89e937663571ae77ccac4430ea6a2e65780b2cde16b8a2f0134c8330
- user template sha256: 039f46669484157a41447099e13d169f85dbf6ac62321fd4f355cbdae73d8d66
- dataset sha256[:12]: f5a4fe704b5b

## Deviations / confounds

1-2. None for the prompt: the user message is the paper's template verbatim, ending at "Answer:".
6. Prompt variant p_mc: the ONLY change to the paper's system prompt is the canonical unit line
   "cpu_capacity=cores;" -> "cpu_capacity=millicores;", so the requested unit matches the gold labels.
4. Qwen's native thinking mode is disabled (chat template enable_thinking=False) so the only reasoning is the
   prompted CoT.
3. Backend is a local vLLM model, not an API: no system_fingerprint (logged as null); latency_s is the
   batch wall time divided by batch size, not a per-request latency.
5. Strict S0 scoring: no unit conversion or tolerance. Gold stores cpu_capacity in millicores while the paper's
   prompt asks for cores, so correct cpu_capacity answers count as FP+FN. Gold highest/lowest values are taken
   layer-filtered in the real records but globally in the synthetic ones.

## System prompt (zs_cot_p_mc_system.txt)

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

## User message template (zs_p_paper_user.txt)

```
Input:
{
 "service": "{service}",
 "expression": "{expression}",
 "snapshot": {snapshot}
}

Answer:
```
