#!/bin/bash
# Rebuild every result file, table and figure of the midterm report from the archived runs (no GPU needed).
# usage: bash scripts/reproduce_report_material.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/runs results figures tables
for gz in runs/*/predictions.jsonl.gz; do
  d=outputs/runs/$(basename "$(dirname "$gz")")
  mkdir -p "$d"
  [ -f "$d/predictions.jsonl" ] || gunzip -c "$gz" > "$d/predictions.jsonl"
done
python3 scripts/make_midterm_results.py      > /dev/null
python3 scripts/make_midterm_errors.py       > /dev/null
python3 scripts/make_prompt_ablation.py      > /dev/null
python3 scripts/make_rejection_results.py    > /dev/null
python3 scripts/make_midterm_tables.py
python3 scripts/make_midterm_figures.py
python3 scripts/make_fs_comparison_figure.py
python3 scripts/verify_report_claims.py     > /dev/null
echo "done: results/*.json (incl. report_claims.json), tables/*.tex, figures/*.pdf"
