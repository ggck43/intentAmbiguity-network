#!/bin/bash
# Rejection experiment: run the 195 invalid records for 4 models x {zs, fs-f2} x {no thinking, thinking}.
# usage: scripts/run_invalid_chain.sh <gpu> <model> <short>
GPU=$1; MODEL=$2; SHORT=$3
cd /home/jovyan/llmExp
PY=.venv-vllm/bin/python
EXTRA=""; case "$MODEL" in *gemma*) EXTRA="--text-only";; esac
FS=fs2_unitFixed+6_prompt.txt
export CUDA_VISIBLE_DEVICES=$GPU
C="--subset invalid --tag invalid --gpu-mem 0.7 $EXTRA --model $MODEL"
$PY scripts/run_zs_cot.py $C --regime zs --prompt-variant p_mc --max-model-len 12288 > outputs/invalid/${SHORT}_zs.log 2>&1
$PY scripts/run_zs_cot.py $C --prompt-file $FS --exp-label fs-f2 --max-model-len 12288 > outputs/invalid/${SHORT}_fsf2.log 2>&1
$PY scripts/run_zs_cot.py $C --regime zs --prompt-variant p_mc --thinking --max-model-len 12288 > outputs/invalid/${SHORT}_zs_think.log 2>&1
$PY scripts/run_zs_cot.py $C --prompt-file $FS --exp-label fs-f2 --thinking --max-model-len 16384 > outputs/invalid/${SHORT}_fsf2_think.log 2>&1
touch outputs/invalid/${SHORT}.done
