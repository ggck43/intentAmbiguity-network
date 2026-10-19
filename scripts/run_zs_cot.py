"""Zero-shot (--regime zs) or zero-shot CoT (--regime zs_cot, default) generation with a local HF model on vLLM.

Examples
  python scripts/run_zs_cot.py --print-prompts 2            # render prompts + token counts (tokenizer only)
  python scripts/run_zs_cot.py --limit 5                    # smoke test on 5 records, own run dir
  python scripts/run_zs_cot.py                              # full CoT run on the 521 valid records
  python scripts/run_zs_cot.py --regime zs                  # full zero-shot (no CoT) run
  python scripts/run_zs_cot.py --resume outputs/runs/<dir>  # continue; skips indices already written
Then: python -m src.score outputs/runs/<dir>
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# FlashInfer's JIT-compiled sampling kernel failed to build here (ninja/nvcc); greedy decoding doesn't need it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import DATASET_PATH, ROOT, SNAPSHOT_PATH, load_records, load_snapshot, snapshot_compact  # noqa: E402
from src.parse import parse_response  # noqa: E402
from src.fewshot import FS1_NAME, FS2_NAME  # noqa: E402
from src.prompts import (REGIMES, SYSTEM_PATHS, USER_PATHS, VARIANTS, build_messages, load_fs_prompt,  # noqa: E402
                         load_system, load_user_template, sha256)

COT_DEVIATIONS = """\
1. The paper's Fig. 8 caption only says to put the CoT phrase into the template. Here "Let's think step-by-step."
   is the last line of the user message (after "Answer:"), preceded on its own line by "Reason first, then give
   the final answer as one JSON object matching the schema above." The same wording is used for every record.
2. The system prompt says "Output MUST be JSON only" while CoT asks for reasoning first. Left as is (known confound).
"""
NOTHINK_DEVIATIONS = """\
4. Qwen's native thinking mode is disabled (chat template enable_thinking=False) so the only reasoning is the
   prompted CoT.
"""
FORCE_TEXT = "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n"
ZS_DEVIATIONS = """\
1-2. None for the prompt: the user message is the paper's template verbatim, ending at "Answer:".
"""
MC_DEVIATIONS = """\
6. Prompt variant p_mc: the ONLY change to the paper's system prompt is the canonical unit line
   "cpu_capacity=cores;" -> "cpu_capacity=millicores;", so the requested unit matches the gold labels.
"""
FS_DEVIATIONS = """\
1-2. Few-shot prompt file (design and justification: notebook 3.fs_promptDesign.ipynb). The examples sit in the single
   user message (paper format); the paper's three system-prompt header lines (Service / Intent Expression / Snapshot
   placeholders) are omitted, as in every zero-shot run. fs1 keeps the paper's system prompt (cores) and its decimal-GB
   example; fs2 states millicores and binary data sizes and has a mandatory snapshot-verification rule.
"""
GEMMA_DEVIATIONS = """\
7. Gemma 4 (instruction-tuned) is run through its own chat template with a normal system turn. Thinking is switched with
   enable_thinking (the template puts <|think|> into the system turn). Greedy decoding is used for comparability with the
   Qwen runs although the model card recommends temperature 1.0, top_p 0.95, top_k 64. Thinking is budgeted in two stages
   as for Qwen: stage 1 stops at <channel|> (token id of the end-of-thought marker) or after --think-budget tokens; if the
   thought was not closed the standard budget-forcing sentence plus <channel|> is appended; stage 2 generates the answer.
   Only the stage-2 text is parsed and scored. Without thinking the model answers directly.
"""
THINK_DEVIATIONS = """\
4. Qwen's native thinking mode is ENABLED (chat template default, generation prompt ends with "<think>\\n").
   Two-stage generation with a thinking budget: stage 1 generates up to --think-budget tokens and stops at
   "</think>". If the model did not close the block, the thinking is cut and the standard budget-forcing text
   is appended before "</think>". Stage 2 generates the answer (up to --answer-tokens) after "</think>".
   Only the stage-2 text is parsed and scored; the thinking text is stored separately (think_text).
"""
COMMON_DEVIATIONS = """\
3. Backend is a local vLLM model, not an API: no system_fingerprint (logged as null); latency_s is the
   batch wall time divided by batch size, not a per-request latency.
5. Strict S0 scoring: no unit conversion or tolerance. Gold stores cpu_capacity in millicores while the paper's
   prompt asks for cores, so correct cpu_capacity answers count as FP+FN. Gold highest/lowest values are taken
   layer-filtered in the real records but globally in the synthetic ones.
"""


class GpuEnergyMeter:
    """GPU-only energy of the visible devices: NVML total-energy counter (mJ) over the generation window,
    plus a 0.5 s power trace for mean/peak power. Host CPU/RAM/cooling are NOT included."""

    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            self.ids = [int(x) for x in vis.split(",")] if vis else list(range(pynvml.nvmlDeviceGetCount()))
            self.nv = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.ids]
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handles[0])
            self.ok = True
        except Exception as e:  # no NVML / UUID-style CUDA_VISIBLE_DEVICES: run without energy numbers
            print(f"energy meter disabled: {e}")

    def _power_w(self):
        return sum(self.nv.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000.0

    def _energy_j(self):
        return sum(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles) / 1000.0

    def idle_power_w(self, seconds=5.0, step=0.5):
        """Mean power with the model loaded and no requests running."""
        if not self.ok:
            return None
        vals = []
        for _ in range(int(seconds / step)):
            vals.append(self._power_w())
            time.sleep(step)
        return sum(vals) / len(vals)

    def start(self, step=0.5):
        if not self.ok:
            return
        self._trace, self._stop = [], threading.Event()

        def sample():
            while not self._stop.is_set():
                self._trace.append(self._power_w())
                self._stop.wait(step)

        self._e0, self._t0 = self._energy_j(), time.time()
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.ok:
            return None
        e1, t1 = self._energy_j(), time.time()
        self._stop.set()
        self._thread.join()
        return {"gpu_energy_J": e1 - self._e0, "gen_seconds": t1 - self._t0,
                "mean_power_W": sum(self._trace) / len(self._trace) if self._trace else None,
                "peak_power_W": max(self._trace) if self._trace else None,
                "devices": self.ids, "gpu_name": self.gpu_name}


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def model_revision(model):
    try:
        from huggingface_hub import model_info
        return model_info(model).sha
    except Exception:
        return None


def _vllm_version():
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:
        return None


def select_records(subset, limit):
    recs = load_records()
    if subset == "valid":
        recs = [r for r in recs if r["meta"]["status"] == "valid"]
    elif subset == "invalid":
        recs = [r for r in recs if r["meta"]["status"] != "valid"]
    return recs[:limit] if limit else recs


def render_chat(tok, messages, thinking=False):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def done_indices(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["index"] for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--regime", choices=REGIMES, default="zs_cot")
    ap.add_argument("--prompt-file", choices=[FS1_NAME, FS2_NAME], default=None,
                    help="few-shot prompt file in prompts/ (replaces --regime / --prompt-variant)")
    ap.add_argument("--expr-file", default=None,
                    help="JSON {index: expression} replacing the intent expressions (template-shift probe; gold labels unchanged)")
    ap.add_argument("--exp-label", default=None, help="experiment label for the run dir and predictions, e.g. fs-f1 (with --prompt-file)")
    ap.add_argument("--prompt-variant", choices=VARIANTS, default="p_paper",
                    help="p_mc = paper system prompt with cpu_capacity=millicores (matches gold)")
    ap.add_argument("--subset", choices=["valid", "invalid", "all"], default="valid")
    ap.add_argument("--limit", type=int, default=0, help="only the first N selected records (smoke test)")
    ap.add_argument("--print-prompts", type=int, default=0, help="print N rendered prompts and exit")
    ap.add_argument("--resume", default=None, help="existing run dir to continue")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--thinking", action="store_true",
                    help="enable Qwen native thinking (two-stage, budgeted); run is labelled <regime>_think")
    ap.add_argument("--think-budget", type=int, default=4096, help="max thinking tokens (stage 1)")
    ap.add_argument("--answer-tokens", type=int, default=1024, help="max answer tokens after </think> (stage 2)")
    ap.add_argument("--tag", default="", help="extra run-dir tag (e.g. energy) so the dir is not picked up by the default run globs")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64, help="records per vLLM call (= resume granularity)")
    ap.add_argument("--gpu-mem", type=float, default=0.5, help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="vLLM max concurrent sequences (default: vLLM's 1024; lower it if startup fails with a Mamba-cache error)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--text-only", action="store_true", help="multimodal models: allow no images/audio/video (skips their encoders)")
    a = ap.parse_args()
    is_gemma = "gemma" in a.model.lower()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    snap = load_snapshot()
    snap_str = snapshot_compact(snap)
    fs = bool(a.prompt_file)
    if fs:
        system, template = load_fs_prompt(a.prompt_file)
        base_label = a.exp_label or "fs"
    else:
        system, template = load_system(a.prompt_variant), load_user_template(a.regime)
        base_label = a.regime
    label = f"{base_label}_think" if a.thinking else base_label
    vtag = ("" if (fs or a.prompt_variant == "p_paper") else f"_{a.prompt_variant}") + (f"_{a.tag}" if a.tag else "")
    recs = select_records(a.subset, a.limit)
    if a.expr_file:
        import copy
        mapping = json.loads(Path(a.expr_file).read_text())
        recs = [copy.deepcopy(r) for r in recs]
        for r in recs:
            r["input"]["intent"]["expression"] = mapping[str(r["index"])]

    if a.print_prompts:
        print(f"snapshot: {len(snap_str)} chars, {len(tok(snap_str)['input_ids'])} tokens ({a.model} tokenizer)")
        for r in recs[:a.print_prompts]:
            msgs = build_messages(r, snap_str, system, template)
            text = render_chat(tok, msgs, a.thinking)
            print(f"\n{'=' * 30} record {r['index']} | {len(tok(text)['input_ids'])} prompt tokens {'=' * 30}")
            print(text.replace(snap_str, "<SNAPSHOT %d chars>" % len(snap_str)))
        return

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        sys.exit("vllm is not installed. Install it in a fresh venv (it pins its own torch): pip install vllm")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = "_smoke" if a.limit else ""
    run_dir = Path(a.resume) if a.resume else ROOT / "outputs" / "runs" / f"{label}{vtag}_{a.model.split('/')[-1].lower().replace('.', '')}{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    done = done_indices(pred_path)
    todo = [r for r in recs if r["index"] not in done]
    print(f"run dir: {run_dir}\nselected {len(recs)} | already done {len(done)} | to do {len(todo)}")
    if not todo:
        return

    sys_hash, user_hash = sha256(system), sha256(template)
    dataset_version = sha256(DATASET_PATH.read_text())[:12]
    revision = model_revision(a.model)
    if not a.resume:
        (run_dir / "README.md").write_text(
            f"# {('Few-shot ' + a.prompt_file) if fs else ('Zero-shot CoT' if a.regime == 'zs_cot' else 'Zero-shot')}{' + native thinking' if a.thinking else ''} ({label}), {a.model}\n\n- date: {dt.datetime.now().isoformat(timespec='seconds')}\n"
            f"- git commit: {git_hash()}\n- model revision: {revision}\n- subset: {a.subset} ({len(recs)} records)\n- expression file: {a.expr_file or 'none (original expressions)'}\n"
            f"- thinking: {a.thinking}\n- prompt variant: {a.prompt_file or a.prompt_variant}\n- max_num_seqs: {a.max_num_seqs or 'vLLM default'}, gpu_memory_utilization: {a.gpu_mem}\n"
            + (f"- decoding: temperature=0, top_p=1, think budget={a.think_budget}, answer max_tokens={a.answer_tokens}, seed={a.seed}\n"
               if a.thinking else f"- decoding: temperature=0, top_p=1, max_tokens={a.max_tokens}, seed={a.seed}\n") +
            f"- vLLM: {_vllm_version()}, batch={a.batch} records per call, max_model_len={a.max_model_len}"
            " (greedy outputs are reproducible for the same batch size, not across batch sizes)\n"
            f"- snapshot: {SNAPSHOT_PATH.name} ({len(tok(snap_str)['input_ids'])} tokens)\n"
            f"- system prompt sha256: {sys_hash}\n- user template sha256: {user_hash}\n- dataset sha256[:12]: {dataset_version}\n\n"
            f"## Deviations / confounds\n\n{FS_DEVIATIONS if fs else (COT_DEVIATIONS if a.regime == 'zs_cot' else ZS_DEVIATIONS)}{MC_DEVIATIONS if (not fs and a.prompt_variant == 'p_mc') else ''}{GEMMA_DEVIATIONS if is_gemma else ''}{THINK_DEVIATIONS if (a.thinking and not is_gemma) else (NOTHINK_DEVIATIONS if not is_gemma else '')}{COMMON_DEVIATIONS}\n"
            f"## System prompt ({a.prompt_file if fs else SYSTEM_PATHS[a.prompt_variant].name})\n\n```\n{system}\n```\n\n"
            f"## User message template ({a.prompt_file if fs else USER_PATHS[a.regime].name})\n\n```\n{template}\n```\n")

    extra = {"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}
    if a.text_only:
        extra["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, seed=a.seed,
              gpu_memory_utilization=a.gpu_mem, tensor_parallel_size=a.tp, enable_prefix_caching=True, **extra)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed)
    sp_think = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.think_budget, seed=a.seed, stop=["</think>"])
    sp_answer = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.answer_tokens, seed=a.seed)
    run_id = run_dir.name

    def generate_thinking(msgs):
        """Two-stage budgeted thinking. Returns per record: answer text/output, think text, flags."""
        prompts = [render_chat(tok, m, True) for m in msgs]
        assert all(p.endswith("<think>\n") for p in prompts), "chat template did not open a think block"
        t_outs = llm.generate(prompts, sp_think, use_tqdm=False)
        stage2, meta = [], []
        for p, o in zip(prompts, t_outs):
            c = o.outputs[0]
            closed = c.finish_reason == "stop" and c.stop_reason == "</think>"
            think = c.text.rstrip()
            tail = "\n</think>\n\n" if closed else FORCE_TEXT + "</think>\n\n"
            stage2.append(p + think + tail)
            meta.append({"think_text": c.text, "think_tokens": len(c.token_ids), "think_closed": closed})
        a_outs = llm.generate(stage2, sp_answer, use_tqdm=False)
        return a_outs, meta

    if is_gemma:
        ch_end = tok.convert_tokens_to_ids("<channel|>")
        sp_think = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.think_budget, seed=a.seed,
                                  stop_token_ids=[ch_end], skip_special_tokens=False)

    def generate_plain_gemma(msgs):
        return llm.generate([render_chat(tok, m, False) for m in msgs], sp, use_tqdm=False)

    def generate_thinking_gemma(msgs):
        """Two-stage budgeted thinking for Gemma 4: <|channel>thought ... <channel|> answer."""
        prompts = [render_chat(tok, m, True) for m in msgs]
        t_outs = llm.generate(prompts, sp_think, use_tqdm=False)
        outs, meta, todo2, stage2 = [None] * len(msgs), [], [], []
        for k, (p, o) in enumerate(zip(prompts, t_outs)):
            c = o.outputs[0]
            closed = c.finish_reason == "stop" and c.stop_reason == ch_end
            direct = c.finish_reason == "stop" and not closed          # the model ended its turn without a thought block
            meta.append({"think_text": "" if direct else c.text, "think_tokens": len(c.token_ids), "think_closed": closed or direct})
            if direct:
                outs[k] = o
            else:
                tail = "\n<channel|>" if closed else FORCE_TEXT + "<channel|>"
                todo2.append(k); stage2.append(p + c.text.rstrip() + tail)
        if stage2:
            for k, o in zip(todo2, llm.generate(stage2, sp_answer, use_tqdm=False)):
                outs[k] = o
        return outs, meta

    meter = GpuEnergyMeter()
    idle_w = meter.idle_power_w()
    meter.start()
    with open(pred_path, "a") as out:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            msgs = [build_messages(r, snap_str, system, template) for r in chunk]
            t0 = time.time()
            if a.thinking:
                outs, tmeta = (generate_thinking_gemma if is_gemma else generate_thinking)(msgs)
            elif is_gemma:
                outs, tmeta = generate_plain_gemma(msgs), [{}] * len(chunk)
            else:
                outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
                tmeta = [{}] * len(chunk)
            per = (time.time() - t0) / len(chunk)
            for r, m, o, tm in zip(chunk, msgs, outs, tmeta):
                c = o.outputs[0]
                p = parse_response(c.text)  # answer text only: thinking text never reaches the parser
                out.write(json.dumps({
                    "run_id": run_id, "model": a.model, "model_revision": revision, "regime": label,
                    "thinking": a.thinking,
                    "prompt_variant": a.prompt_file or a.prompt_variant, "dataset_version": dataset_version,
                    "system_prompt_hash": sys_hash, "user_prompt_hash": sha256(m[1]["content"]),
                    "index": r["index"], "raw_text": c.text, "parsed": p["parsed"], "parse_ok": p["parse_ok"],
                    "n_json_objects": p["n_objects"], "chose_last_object": p["chose_last"],
                    "truncated": c.finish_reason == "length", "finish_reason": c.finish_reason,
                    **tm,
                    "forced_close": (not tm["think_closed"]) if a.thinking else None,
                    "system_fingerprint": None, "latency_s": round(per, 3),
                    "prompt_tokens": len(o.prompt_token_ids) if not a.thinking else None,
                    "completion_tokens": len(c.token_ids) + tm.get("think_tokens", 0),
                    "answer_tokens": len(c.token_ids),
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                }) + "\n")
            out.flush()
            print(f"  {min(i + a.batch, len(todo))}/{len(todo)} done ({per:.2f}s/record)")
    energy = meter.stop()
    if energy:
        energy.update(idle_loaded_W=idle_w, n_records=len(todo), model=a.model, regime=label,
                      prompt_variant=a.prompt_variant, tensor_parallel=a.tp, batch=a.batch)
        (run_dir / "energy.json").write_text(json.dumps(energy, indent=2) + "\n")
        print(f"GPU energy: {energy['gpu_energy_J'] / 3600:.2f} Wh in {energy['gen_seconds']:.0f} s "
              f"(mean {energy['mean_power_W']:.0f} W, idle-loaded {idle_w:.0f} W)")
    print(f"\nscore with: python -m src.score {run_dir}")


if __name__ == "__main__":
    main()
