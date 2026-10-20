# Lightweight LLMs for Natural-Language Intent-to-SLO Extraction

Course project for Deep Learning (School of Computing, Queen's University), by Sabbir Ahmed.

An operator writes a placement request in plain language, for example:

> *"Place the recommender on the edge, keep CPU utilization below 55%, and use the node with the lowest memory utilization."*

The system has to turn it into a JSON list of service-level objectives (SLOs) that a scheduler can use. Some values
are written in the request. Others ("the node with the lowest ...") must be looked up in a snapshot of the cluster.

This repository tests how well **small open models on one GPU** can do this. It also tests how their results can be
improved with prompt design and thinking mode. We use the dataset and the scoring rules of the
[Intent Engine](https://doi.org/10.1016/j.sysarc.2026.103938) paper (Islam & Calheiros, *J. Syst. Archit.* 179, 2026;
data on [Zenodo](https://doi.org/10.5281/zenodo.20810799)). We do **not** re-implement the Intent Engine pipeline
(parser, validation algorithm, retrieval). Every result here comes from plain prompting.

## Why open lightweight models

The paper's numbers come from closed-source models that need paid API access, and these models are now older. So we
use open models that run locally: **Qwen3.5 (4B, 9B)** and **Gemma 4 (E2B, E4B)**. We cannot rerun the paper's models, so
we test two sizes of each family instead. A finding that holds across two sizes and two families is less likely to be
an accident of one model.

## Main results (midterm)

Strict micro-F1 on the 521 valid intents (exact match on type, metric, comparator, tag, value and unit). Single greedy runs.

| Model | Zero-shot | Zero-shot + thinking | Few-shot (fs-f2) | Few-shot + thinking |
|---|---|---|---|---|
| Qwen3.5-4B | 0.348 | 0.543 | 0.854 | **0.953** |
| Qwen3.5-9B | 0.828 | 0.833 | 0.925 | 0.948 |
| Gemma 4 E2B | 0.239 | 0.385 | 0.645 | 0.902 |
| Gemma 4 E4B | 0.793 | 0.831 | 0.877 | 0.939 |

**Baseline check.** With the paper's own prompts, Qwen3.5-4B scores 0.334 zero-shot (paper, GPT-4.1 mini: 0.331) and
0.735 few-shot (paper: 0.751).

**Prompt ablation (Qwen, thinking off, total F1 4B / 9B).** Paper zero-shot prompt 0.334 / 0.761; with the paper's CoT
phrase 0.380 / 0.760; one word changed (cores → millicores) 0.348 / 0.828; paper's few-shot examples (fs1) 0.735 / 0.887;
our few-shot prompt (fs2) 0.854 / 0.925.

**What we found.**
- Many errors are not language errors. They are conventions: CPU capacity in cores instead of millicores, decimal
  instead of binary GB, a `highest` tag added to plain thresholds by habit (512 times for the 4B model with the
  paper's prompt), and lookups left as `null` (109 of 221 for the 9B model in zero-shot).
- A prompt that states these conventions and shows worked lookups (`prompts/fs2_unitFixed+6_prompt.txt`) lifts the 4B
  model from 0.348 to 0.854, and to 0.953 with thinking.
- The best settings reach the ceiling of the dataset. Most remaining "errors" are a label conflict: in 59 synthetic
  records (61 SLOs) the gold uses the global extreme even when the request names a layer. Counting those as correct
  (a diagnostic only) gives 0.999 (Qwen 4B) and 1.000 (Qwen 9B) for few-shot with thinking.
- On the 195 invalid intents, the models rarely refuse without thinking (rejection rate 0.04–0.27). With thinking the
  rate rises to 0.53–0.62. Conflicting requests are almost never rejected (0.00–0.02).

## Repository layout

| Path | Content |
|---|---|
| `src/data_io.py` | dataset and snapshot loaders |
| `src/prompts.py`, `src/fewshot.py` | prompt builders; generator and checks for the few-shot examples |
| `src/parse.py` | takes the last JSON object with a `slos` list out of the model text |
| `src/score.py` | strict micro-F1, exact match, Jaccard, template-clustered bootstrap CI |
| `src/analysis.py` | error taxonomy, per-metric scores, lookup outcomes |
| `src/reject.py` | rejection rate on the invalid intents (Fig. 6 of the paper) |
| `prompts/` | all prompt files (paper zero-shot, millicores variant, few-shot fs1 and fs2) |
| `data/raw/` | the 716 records and the infrastructure snapshot (from the Zenodo release) |
| `scripts/run_zs_cot.py` | generation with vLLM: zero-shot / few-shot, thinking off / on, Qwen and Gemma |
| `scripts/run_invalid_chain.sh` | the four rejection runs for one model |
| `scripts/make_*.py` | result files, tables and figures of the report |
| `scripts/reproduce_report_material.sh` | rebuilds all report material from the archived runs (no GPU) |
| `scripts/verify_report_claims.py` | recomputes the numbers quoted in the report text (habit-tag count, unit conventions, null lookups, scope conflict, ceilings, coverage, re-wording loss) → `results/report_claims.json` |
| `figures/src/pipeline.tex` | TikZ source of the pipeline figure (Fig. 1 of the report) |
| `runs/` | one folder per run: `README.md` (exact settings, prompt hashes) and `predictions.jsonl.gz` (raw text, parsed JSON, thinking text) |
| `results/`, `tables/`, `figures/` | generated report material |
| `notebooks/` | analysis notebooks with saved outputs: 3 prompt design, 4 Qwen few-shot, 6 Gemma, 9 invalid-intent rejection |
| `tests/` | unit tests: dataset facts, parser, scoring, prompts, few-shot examples, rejection |

## Where each part of the midterm report comes from

| Report item | File | Made by |
|---|---|---|
| Fig. 1 (pipeline) | `figures/fig_pipeline.pdf` | `figures/src/pipeline.tex` (TikZ) |
| Table I (baseline check vs the paper) | `tables/tab_baseline.tex` | `make_midterm_results.py` → `make_midterm_tables.py` |
| Table II (main results, Table-7 style) | `tables/tab_main.tex` | `make_midterm_results.py` → `make_midterm_tables.py` |
| Table III (prompt ablation) | `tables/tab_ablation.tex` | `make_prompt_ablation.py` |
| Table IV and Fig. 4 (rejection on invalid intents) | `tables/tab_reject.tex`, `figures/fig_rejection.pdf` | `make_rejection_results.py` |
| Table V (error counts) | `tables/tab_errors.tex` | `make_midterm_errors.py` → `make_midterm_tables.py` |
| Table VI (failure cases) | written by hand from the run files listed in the table | – |
| Fig. 2 (level-wise F1) | `figures/fig_levelwise_f1.pdf` | `make_midterm_figures.py` |
| Fig. 3 (exact match and Jaccard) | `figures/fig_em_jaccard.pdf` | `make_midterm_figures.py` |
| Numbers quoted in the text | `results/report_claims.json` | `verify_report_claims.py` |

`figures/fig_fs_vs_paper.pdf` (our few-shot + thinking against the paper's few-shot CoT) is still generated by
`make_fs_comparison_figure.py` but is not used in the current report.

## Quick start: rebuild the report material (no GPU)

```bash
pip install -r requirements.txt
bash scripts/reproduce_report_material.sh   # unpacks runs/ into outputs/runs/, writes results/, tables/, figures/,
                                            # and results/report_claims.json
python3 -m pytest -q tests                  # 25 tests
```

We checked this in a clean copy of the repository: the generated tables are identical to the ones in the report.

## Running the models (GPU)

Use a separate environment, because vLLM pins its own PyTorch version. We used vLLM 0.31.0 and transformers 5.17.0 on
one NVIDIA H100 (96 GB).

```bash
python -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install vllm==0.31.0 transformers==5.17.0 huggingface_hub
export VLLM_USE_FLASHINFER_SAMPLER=0

# zero-shot (millicores prompt), thinking off / on
python scripts/run_zs_cot.py --model Qwen/Qwen3.5-9B --regime zs --prompt-variant p_mc --gpu-mem 0.7 --max-model-len 12288
python scripts/run_zs_cot.py --model Qwen/Qwen3.5-9B --regime zs --prompt-variant p_mc --thinking --gpu-mem 0.7 --max-model-len 12288

# few-shot (fs-f2), thinking off / on
python scripts/run_zs_cot.py --model Qwen/Qwen3.5-9B --prompt-file fs2_unitFixed+6_prompt.txt --exp-label fs-f2 --gpu-mem 0.7 --max-model-len 12288
python scripts/run_zs_cot.py --model Qwen/Qwen3.5-9B --prompt-file fs2_unitFixed+6_prompt.txt --exp-label fs-f2 --thinking --gpu-mem 0.7 --max-model-len 16384

# Gemma: --model google/gemma-4-E4B-it --text-only
# invalid intents: add --subset invalid --tag invalid, or: bash scripts/run_invalid_chain.sh <gpu> <model> <short-name>
# paper prompts: --regime zs (cores), --regime zs_cot, --prompt-file fs1_verbatim+3_prompt.txt --exp-label fs-f1

python3 -m src.score outputs/runs/<run-dir>   # writes scores.json for one run
```

Each run writes `outputs/runs/<regime>_<model>_<timestamp>/` with `predictions.jsonl` and a `README.md` that lists
every setting.

## Settings and reproducibility

- Greedy decoding (temperature 0, top_p 1), seed 0, bf16, 64 records per vLLM call. Greedy outputs are reproducible
  for the same batch size.
- Thinking: up to 4,096 thinking tokens, then a forced close with a fixed sentence, then up to 1,024 answer tokens. Only
  the answer is parsed and scored.
- Zero-shot without thinking: up to 2,048 output tokens.
- One run per setting. There are no confidence intervals in the midterm numbers yet.

## Metrics

Every SLO becomes a tuple *(type, metric, comparator, tag, value, unit)*. Two SLOs match only if all six fields are
equal, with no tolerance on the value. TP, FP and FN are summed over all records (micro average). Exact match and
Jaccard are computed per record and averaged. For invalid intents, a response with no SLO counts as a rejection.

## Known limitations

- The few-shot prompt was designed by looking at errors on the same 521 records. Its examples use invented services,
  and a re-worded version of the intents keeps the scores, but there is no held-out split yet.
- The dataset uses one snapshot and very regular language (62 word types cover all valid intents).
- Strict matching counts equal values in different units as errors, by design (as in the paper).

## Citation of the dataset and baseline

K. Islam and R. N. Calheiros, "Intent Engine: Natural-language intent translation for intent-driven orchestration in
the compute continuum," *Journal of Systems Architecture*, vol. 179, p. 103938, 2026.
