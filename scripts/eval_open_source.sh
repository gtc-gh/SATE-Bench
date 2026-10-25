#!/bin/bash
# Active + Passive evaluation of the open-source models in the paper (vLLM; one GPU per model).
# GEMINI_API_KEY is optional: it enables the LLM-based answer extraction fallback.
# Gemma is a gated model: accept its license on Hugging Face and run `hf auth login` first.
set -e
cd "$(dirname "$0")/.."

for name in qwen2.5-vl-3b-instruct qwen3-vl-8b-instruct qwen3-vl-8b-thinking gemma-3-4b-it llava-onevision-2-8b; do
    python evaluation/run_active.py  --config "configs/models/${name}.yaml"
    python evaluation/run_passive.py --config "configs/models/${name}.yaml"
done
