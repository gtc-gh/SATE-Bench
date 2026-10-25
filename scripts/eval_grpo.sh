#!/bin/bash
# Evaluate a GRPO checkpoint (LoRA adapter) in the Active and Passive settings.
#   bash scripts/eval_grpo.sh checkpoints/grpo_qwen25vl_3b
set -e
cd "$(dirname "$0")/.."

CKPT=${1:?usage: bash scripts/eval_grpo.sh <checkpoint_dir> [base_model]}
BASE_MODEL=${2:-Qwen/Qwen2.5-VL-3B-Instruct}
NAME=$(basename "${CKPT}")

python evaluation/run_active_grpo.py --lora_path "${CKPT}" --model "${BASE_MODEL}"
python evaluation/run_passive.py --config configs/models/grpo-qwen2.5-vl-3b.yaml \
    --model "${BASE_MODEL}" --lora_path "${CKPT}" --results_dir "results/passive/grpo/${NAME}"
