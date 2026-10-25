#!/bin/bash
# ==============================================================================
# SATE-Bench GRPO training launcher.
#
# All hyper-parameters live in training/config.yaml (Table 20 of the paper);
# any of them can be overridden on the command line.
#
#   bash training/train.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/train.sh
#   bash training/train.sh --action_weight 1.2 --output_dir checkpoints/grpo_qwen25vl_3b_action1.2
#   bash training/train.sh --report_to none
#
# Weights & Biases: run `wandb login` once (or export WANDB_API_KEY) before training.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-SATE-Bench-GRPO}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS=${#GPUS[@]}
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    NUM_GPUS=${NUM_GPUS:-1}
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --master_port=$((29500 + RANDOM % 1000))"
else
    LAUNCHER="python3"
fi

echo "Launching GRPO training on ${NUM_GPUS} GPU(s) with ${SCRIPT_DIR}/config.yaml"
${LAUNCHER} training/train_grpo.py --config "${SCRIPT_DIR}/config.yaml" "$@"
