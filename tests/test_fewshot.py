import json
import re

import pytest

from src import fewshot as F
from src.data_io import load_records, load_snapshot, snapshot_compact
from src.parse import parse_response
from src.prompts import build_messages_fs, load_fs_prompt
from src.score import canon_list


def _norm(expr, service):
    return re.sub(r"\d+(\.\d+)?", "N", expr.replace(service, "<SVC>")).strip().rstrip(".?")


def test_prompt_files_match_generator():
    for name in (F.FS1_NAME, F.FS2_NAME):
        assert (F.PROMPT_DIR / name).read_text(encoding="utf-8") == F.file_text(name)


def test_loader_roundtrip_and_placeholders():
    for name in (F.FS1_NAME, F.FS2_NAME):
        system, user = load_fs_prompt(name)
        assert system == F.build(name)[0].rstrip("\n") or system == F.build(name)[0].rstrip()
        assert len(re.findall(r"^Example \d$", user, re.M)) == 6
        for ph in ("{service}", "{expression}", "{snapshot}"):
            assert user.count(ph) == 1
        assert user.rstrip().endswith("Answer:")


def test_rendering_a_real_record_leaves_no_placeholder():
    rec = next(r for r in load_records() if r["meta"]["status"] == "valid")
    snap = snapshot_compact(load_snapshot())
    for name in (F.FS1_NAME, F.FS2_NAME):
        msgs = build_messages_fs(rec, snap, name)
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        assert rec["input"]["intent"]["expression"] in msgs[1]["content"] and snap in msgs[1]["content"]
        for ph in ("{service}", "{expression}", "{snapshot}"):
            assert ph not in msgs[1]["content"]


def test_example_outputs_parse_and_match_their_spec():
    for k, ex in F.EXAMPLES.items():
        out = F.render_output(ex)
        p = parse_response(out)
        assert p["parse_ok"] and p["parsed"]["slos"] == ex["slos"], k
        assert canon_list(ex["slos"])  # canonicalises without error
    for k, block in F.PAPER_EXAMPLES.items():
        assert parse_response(block.split("Output:\n", 1)[1])["parse_ok"], k


def test_implicit_values_come_from_the_right_nodes():
    e = F.EXAMPLES
    v = lambda k, i: e[k]["slos"][i]["target"]["value"]
    assert v("M4", 2) == 4.1 and v("M5", 1) == 4125683712 and v("M5", 2) == 0.047
    assert v("M6", 2) == 23.0 and v("M6", 3) == 62220779520
    for k in ("M4", "M5", "M6"):
        layer = next(x for x in e[k]["slos"] if x["type"] == "placement_location")["target"]["value"]
        traps = 0
        for s in e[k]["slos"]:
            if s["target"]["op"] in ("highest", "lowest"):
                # the layer-restricted answer is what is demonstrated ...
                assert F.implicit(e[k]["snapshot"], s["metric"], s["target"]["op"], layer)["target"]["value"] == s["target"]["value"]
                # ... and the unrestricted one differs for at least one lookup of each example (layer trap)
                traps += F.implicit(e[k]["snapshot"], s["metric"], s["target"]["op"])["target"]["value"] != s["target"]["value"]
        assert traps >= 1, k
    # every lookup example also has a node with null readings that must be skipped
    for k in ("M4", "M6"):
        assert any(n["utilization_now"]["cpu_util_percent"] is None for n in e[k]["snapshot"]["nodes"])


def test_units_in_examples_follow_gold_conventions():
    e = F.EXAMPLES
    assert e["M1"]["slos"][1]["target"]["value"] == 4000 and e["M1"]["slos"][1]["unit"] == "millicores"
    assert e["M1"]["slos"][2]["target"]["value"] == 8 * 2 ** 30
    assert e["M2"]["slos"][0]["target"]["value"] == 100 * 2 ** 30


def test_fs1_is_paper_prompt_plus_three():
    system, user = F.build(F.FS1_NAME)
    assert system == (F.PROMPT_DIR / "zs_cot_p_paper_system.txt").read_text(encoding="utf-8")  # unchanged system prompt
    for k in ("P1", "P2", "P3"):
        assert F.PAPER_EXAMPLES[k] in user
    assert '"value": 2000000000, "range": null' in user       # the paper's decimal-GB example is kept verbatim
    for k in ("M4", "M5", "M6"):
        assert F.render_example(F.EXAMPLES[k]) in user
    for k in ("M4", "M5", "M6"):  # unit-neutral additions: no cpu_capacity SLO and no GB/core conversion in them
        assert all(x["metric"] != "cpu_capacity" for x in F.EXAMPLES[k]["slos"])
        assert not re.search(r"\b(GB|MB|CPU cores)\b", F.EXAMPLES[k]["expression"])


def test_fs2_changes_are_exactly_the_documented_ones():
    s1, s2 = F.system_fs1(), F.system_fs2()
    assert "cpu_capacity=cores;" in s1 and "cpu_capacity=millicores;" in s2 and "cpu_capacity=cores;" not in s2
    assert "SI/IEC" in s1 and "SI/IEC" not in s2
    assert "MANDATORY" in s2 and "MANDATORY" not in s1
    _, user = F.build(F.FS2_NAME)
    assert "2000000000" not in user and "MiB" not in user and '"range"' not in user
    for k in ("M1", "M2", "M3", "M4", "M5", "M6"):
        assert F.render_example(F.EXAMPLES[k]) in user


def test_examples_do_not_copy_dataset_expressions():
    recs = load_records()
    templates = {_norm(r["input"]["intent"]["expression"], r["input"]["intent"]["service"]) for r in recs}
    for k, ex in F.EXAMPLES.items():
        assert _norm(ex["expression"], ex["service"]) not in templates, k
        assert ex["service"].startswith("svc/example-")
