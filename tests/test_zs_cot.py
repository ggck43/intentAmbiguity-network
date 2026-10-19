import json
from collections import Counter

import pytest

from src.data_io import load_records, load_snapshot, snapshot_compact, template_id
from src.parse import parse_response
from src.prompts import build_messages, load_system, load_user_template
from src.score import canon_slo, canon_list, record_counts, prf, score_run


@pytest.fixture(scope="module")
def records():
    return load_records()


# 1. dataset facts ---------------------------------------------------------
def test_dataset_facts(records):
    valid = [r for r in records if r["meta"]["status"] == "valid"]
    assert len(records) == 716
    assert len(valid) == 521
    assert Counter(r["meta"]["source"] for r in records) == {"real": 216, "synthetic": 500}
    assert dict(Counter(r["meta"]["level"] for r in valid)) == {1: 137, 2: 174, 3: 141, 4: 48, 5: 21}
    assert len({r["index"] for r in records}) == 716
    # 525 is the template count over all 716 records; the bootstrap uses the valid subset (487)
    assert len({template_id(r) for r in records}) == 525
    assert len({template_id(r) for r in valid}) == 487
    assert all(r["label"]["slos"] == [] for r in records if r["meta"]["status"] != "valid")


# 2. parser ----------------------------------------------------------------
def test_parser_takes_last_object():
    txt = 'reasoning... {"service":"s","slos":[{"a":1}]} more text {"service":"s","slos":[]}'
    out = parse_response(txt)
    assert out["parse_ok"] and out["parsed"] == {"service": "s", "slos": []}
    assert out["n_objects"] == 2 and out["chose_last"]


def test_parser_fenced_and_braces_in_strings():
    txt = 'Thinking {not json}\n```json\n{"service":"a}b","slos":[{"x":"{"}]}\n```'
    out = parse_response(txt)
    assert out["parse_ok"] and out["parsed"]["service"] == "a}b"


def test_parser_no_json():
    out = parse_response("I think the answer is edge.")
    assert out["parse_ok"] is False and out["parsed"] is None


def test_parser_reports_when_last_object_is_not_chosen():
    out = parse_response('{"service":"s","slos":[]} then {"note": 1}')
    assert out["parse_ok"] and not out["chose_last"]


# 3. canonicalisation --------------------------------------------------------
def _slo(cmp_, op, value=3000, typ="static_compute_resource", metric="cpu_capacity", unit="millicores"):
    return {"metric": metric, "target": {"cmp": cmp_, "op": op, "value": value}, "type": typ, "unit": unit}


def test_canonicalisation_cmp_op_equivalence():
    assert canon_slo(_slo(None, ">=")) == canon_slo(_slo(">=", None))
    assert canon_slo(_slo("<=", "lowest")) == canon_slo(_slo(None, "lowest"))  # lowest defaults to <=
    assert canon_slo(_slo(None, "highest"))[2:4] == (">=", "highest")


def test_canonicalisation_placement_and_garbage():
    p = canon_slo({"metric": "x", "target": {"cmp": None, "op": "set", "value": "Edge"},
                   "type": "placement_location", "unit": "y"})
    assert p == ("placement_location", None, "set", None, "edge", None)
    assert canon_slo("not a dict")[0] == "invalid"            # must not crash
    assert canon_slo(_slo(">=", None, value="lots"))[4] == "lots"


# 4. toy record 25 -----------------------------------------------------------
def test_toy_record_25(records):
    gold = canon_list(next(r for r in records if r["index"] == 25)["label"]["slos"])
    pred_slos = [
        {"metric": None, "target": {"cmp": None, "op": "set", "value": "edge"},
         "type": "placement_location", "unit": None},
        {"metric": "storage_utilization", "target": {"cmp": "<=", "op": None, "value": 65},
         "type": "dynamic_compute_resource", "unit": "percent"},
        {"metric": "cpu_utilization", "target": {"cmp": "<=", "op": "lowest", "value": 4.05},
         "type": "dynamic_compute_resource", "unit": "percent"},
        {"metric": "memory_utilization", "target": {"cmp": "<=", "op": None, "value": 50},
         "type": "dynamic_compute_resource", "unit": "percent"},
    ]
    c = record_counts(canon_list(pred_slos), gold)
    assert (c["tp"], c["fp"], c["fn"]) == (2, 2, 1)
    m = prf(c["tp"], c["fp"], c["fn"])
    assert (round(m["precision"], 3), round(m["recall"], 3), round(m["f1"], 3)) == (0.5, 0.667, 0.571)
    assert c["em"] == 0 and round(c["jaccard"], 3) == 0.4


# 5. unparseable record --------------------------------------------------------
def test_unparseable_record_is_fn_only():
    gold = canon_list([_slo(">=", None), _slo(None, ">=", 5, metric="memory_capacity", unit="bytes")])
    c = record_counts([], gold)
    assert (c["tp"], c["fp"], c["fn"]) == (0, 0, 2) and c["em"] == 0 and c["jaccard"] == 0


# prompt -----------------------------------------------------------------------
def test_prompt_ends_with_cot_and_contains_snapshot(records):
    snap = snapshot_compact(load_snapshot())
    msgs = build_messages(records[24], snap)
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == load_system()
    user = msgs[1]["content"]
    assert user.endswith("Let's think step-by-step.")
    assert "Reason first, then give the final answer as one JSON object matching the schema above.\n" \
           "Let's think step-by-step." in user
    assert user.index("Answer:") < user.index("Reason first")
    assert snap in user and records[24]["input"]["intent"]["expression"] in user
    assert "{service}" not in user and "{snapshot}" not in user
    assert "{expression}" in load_user_template()


def test_zero_shot_prompt_has_no_cot(records):
    snap = snapshot_compact(load_snapshot())
    zs = build_messages(records[24], snap, regime="zs")[1]["content"]
    cot = build_messages(records[24], snap, regime="zs_cot")[1]["content"]
    assert zs.endswith("Answer:") and "step-by-step" not in zs and "Reason first" not in zs
    assert cot.startswith(zs)  # CoT = zero-shot + the two appended lines, nothing else changes
    assert cot[len(zs):] == ("\nReason first, then give the final answer as one JSON object matching the "
                             "schema above.\nLet's think step-by-step.")


# 6. scorer is pure -------------------------------------------------------------
def test_scorer_is_pure_and_perfect_predictions_score_one(records, tmp_path):
    lines = []
    for r in records:
        gold = {"service": r["label"]["service"], "slos": r["label"]["slos"]}
        lines.append({"index": r["index"], "parse_ok": True, "parsed": gold, "truncated": False,
                      "regime": "zs", "model": "org/Test-1B"})
    # make one valid record unparseable and one truncated
    valid_idx = [r["index"] for r in records if r["meta"]["status"] == "valid"]
    lines[valid_idx[0] - 1].update(parse_ok=False, parsed=None)
    lines[valid_idx[1] - 1].update(truncated=True)
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    s1 = score_run(tmp_path, B=200)
    b1 = (tmp_path / "scores.json").read_bytes()
    s2 = score_run(tmp_path, B=200)
    assert b1 == (tmp_path / "scores.json").read_bytes() and s1 == s2
    assert s1["n_valid_scored"] == 521
    assert s1["parse_rate"] == pytest.approx(520 / 521)
    assert s1["trunc_rate"] == pytest.approx(1 / 521)
    assert s1["total"]["fp"] == 0 and s1["total"]["fn"] == len(records[valid_idx[0] - 1]["label"]["slos"])
    assert s1["bootstrap_ci"]["n_templates"] == 487
    assert s1["regime"] == "zs"
    assert "| Test-1B (re-run) |" in (tmp_path / "table7_row.md").read_text()
    assert "zero-shot," in (tmp_path / "table7_row.md").read_text()


# analysis helpers (the notebook's findings rest on these) ---------------------------
def _perturbed_preds(records):
    """gold for odd indices, gold minus its last SLO for even ones, unparseable for every 7th valid record."""
    preds, n = {}, 0
    for r in records:
        if r["meta"]["status"] != "valid":
            continue
        n += 1
        slos = list(r["label"]["slos"])
        if r["index"] % 2 == 0 and len(slos) > 1:
            slos = slos[:-1]
        ok = n % 7 != 0
        preds[r["index"]] = {"index": r["index"], "parse_ok": ok, "truncated": False,
                             "parsed": {"service": "s", "slos": slos} if ok else None}
    return preds


def test_analysis_helpers_agree_with_scorer(records):
    from src import analysis as A
    from src.score import per_record
    preds = _perturbed_preds(records)
    rows = per_record(records, preds)
    tot = [sum(r[k] for r in rows) for k in ("tp", "fp", "fn")]
    # per-metric buckets partition the SLOs -> sums equal the pooled totals
    pm = A.per_metric(records, preds)
    assert [int(pm[k].sum()) for k in ("tp", "fp", "fn")] == tot
    # identity transform reproduces the headline F1; neutralising a field can only help here
    f = A.diagnostic_f1(records, preds, lambda t: t, "x")
    assert f["total"] == pytest.approx(A.f1(*tot))
    assert (A.diagnostic_f1(records, preds, A.drop_ext, "x") >= f - 1e-12).all()
    # comparing a run with itself: zero difference, degenerate CI
    d = A.paired_diff_ci(A.record_frame(records, preds, "a"), A.record_frame(records, preds, "b"), B=200)
    assert (d["diff (b - a)"] == 0).all() and (d["CI low"] == 0).all() and (d["CI high"] == 0).all()
    # taxonomy accounts for every unmatched gold SLO exactly once (FN = missed + wrong-* + unparseable)
    tax = A.error_taxonomy(records, preds)
    assert tax[tax.error != "spurious"]["count"].sum() == tot[2]


def test_extremum_outcomes_classification():
    from src import analysis as A
    rec = {"index": 1, "meta": {"status": "valid", "source": "real", "level": 1},
           "label": {"slos": [_slo("<=", "lowest", 4.0, typ="dynamic_compute_resource", metric="cpu_utilization", unit="percent"),
                              _slo("<=", "lowest", 5.0, typ="dynamic_compute_resource", metric="memory_utilization", unit="percent"),
                              _slo(">=", "highest", 2000, metric="cpu_capacity", unit="millicores"),
                              _slo("<=", "lowest", 7.0, typ="dynamic_compute_resource", metric="storage_utilization", unit="percent")]}}
    pred = [_slo("<=", "lowest", None, typ="dynamic_compute_resource", metric="cpu_utilization", unit="percent"),   # null value
            _slo("<=", "highest", 5.0, typ="dynamic_compute_resource", metric="memory_utilization", unit="percent"),  # wrong op
            _slo(">=", "highest", 2, metric="cpu_capacity", unit="cores")]                                           # right after x1000
    out = A.extremum_outcomes([rec], {1: {"parse_ok": True, "parsed": {"slos": pred}}})
    assert out.to_dict() == {"right op, value null (snapshot not read)": 1,
                             "wrong op (e.g. threshold / opposite extreme)": 1,
                             "right op, correct number": 1, "metric not predicted": 1}
