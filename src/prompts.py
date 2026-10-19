"""Prompt construction for the zero-shot (zs) and zero-shot CoT (zs_cot) regimes, variant p_paper.

Both regimes share the paper's system prompt; they differ only in the user template:
zs ends at "Answer:", zs_cot appends the two CoT lines after it.
"""
import hashlib
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
SYSTEM_PATHS = {  # prompt variant -> system prompt; p_mc differs from p_paper only in cpu_capacity=millicores
    "p_paper": PROMPT_DIR / "zs_cot_p_paper_system.txt",
    "p_mc": PROMPT_DIR / "zs_cot_p_mc_system.txt",
}
VARIANTS = tuple(SYSTEM_PATHS)
SYSTEM_PATH = SYSTEM_PATHS["p_paper"]  # shared by both regimes
USER_PATHS = {
    "zs_cot": PROMPT_DIR / "zs_cot_p_paper_user.txt",
    "zs": PROMPT_DIR / "zs_p_paper_user.txt",
}
USER_PATH = USER_PATHS["zs_cot"]
REGIMES = tuple(USER_PATHS)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_system(variant="p_paper"):
    return SYSTEM_PATHS[variant].read_text(encoding="utf-8")


def load_user_template(regime="zs_cot"):
    return USER_PATHS[regime].read_text(encoding="utf-8")


def render_user(template, service, expression, snapshot_str):
    # plain replace, not str.format: the template is full of literal JSON braces
    return (template.replace("{service}", service)
                    .replace("{expression}", expression)
                    .replace("{snapshot}", snapshot_str))


def build_messages(record, snapshot_str, system=None, template=None, regime="zs_cot"):
    system = load_system() if system is None else system
    template = load_user_template(regime) if template is None else template
    intent = record["input"]["intent"]
    user = render_user(template, intent["service"], intent["expression"], snapshot_str)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---- few-shot prompts: one file per prompt (system + user template), built by src/fewshot.py ----
FS_SYSTEM_MARK, FS_USER_MARK = "=== SYSTEM ===", "=== USER ==="


def load_fs_prompt(name):
    """(system, user_template) from a prompts/fs*_prompt.txt file."""
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    head, user = text.split(f"\n{FS_USER_MARK}\n", 1)
    assert head.startswith(FS_SYSTEM_MARK + "\n")
    return head[len(FS_SYSTEM_MARK) + 1:], user.rstrip("\n")


def build_messages_fs(record, snapshot_str, name):
    system, template = load_fs_prompt(name)
    intent = record["input"]["intent"]
    user = render_user(template, intent["service"], intent["expression"], snapshot_str)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
