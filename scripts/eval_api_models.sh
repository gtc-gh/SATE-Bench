#!/bin/bash
# Active + Passive evaluation of the proprietary models in the paper.
# Requires GEMINI_API_KEY (also used for the answer-extraction fallback) and OPENAI_API_KEY.
set -e
cd "$(dirname "$0")/.."

for name in gemini-3.5-flash gemini-3.1-flash-lite gemini-2.5-flash gemini-2.5-flash-lite gpt-5.2 gpt-5-mini gpt-5-nano; do
    python evaluation/run_active.py  --config "configs/models/${name}.yaml"
    python evaluation/run_passive.py --config "configs/models/${name}.yaml"
done
