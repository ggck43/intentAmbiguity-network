"""Prompt token counts with a model's chat template (run with the .venv-vllm interpreter).

Usage: .venv-vllm/bin/python scripts/count_tokens.py Qwen/Qwen3.5-4B < messages.json
stdin: JSON list of chat-message lists; stdout: JSON list of prompt token counts (generation prompt included,
Qwen thinking disabled, which does not change the length of the prompt text itself).
"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
batches = json.load(sys.stdin)
out = []
for msgs in batches:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out.append(len(tok(text)["input_ids"]))
print(json.dumps(out))
