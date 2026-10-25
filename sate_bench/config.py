"""Global constants and paths for SATE-Bench."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BENCHMARK_ROOT = REPO_ROOT / "data" / "benchmark"
ACTIVE_ANNOTATIONS = BENCHMARK_ROOT / "spatial_active.json"
PASSIVE_ANNOTATIONS = BENCHMARK_ROOT / "spatial_passive.json"

TRAIN_ROOT = REPO_ROOT / "data" / "train"
TRAIN_ANNOTATIONS = TRAIN_ROOT / "train.json"
VAL_ANNOTATIONS = TRAIN_ROOT / "val.json"

RESULTS_ROOT = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Evaluation protocol (Sec. 3.4 / 4.1 of the paper)
# ---------------------------------------------------------------------------
MAX_TURN = 8                      # maximum number of views a model may observe in the Active setting
LAMBDA_VAL = 0.1                  # step penalty of the exploration reward R = I(correct) * exp(-lambda * steps)
IMAGE_RESOLUTION = (1280, 720)    # every view is resized to 720p before being sent to a model
API_MAX_RETRIES = 5               # retries for transient API errors
API_RETRY_SLEEP = 10              # seconds between API retries

# Lightweight LLM used to parse irregular outputs when rule-based parsing fails.
EXTRACTOR_MODEL = "gemini-2.5-flash-lite"
EXTRACTOR_MAX_RETRIES = 3

TASKS = [
    "Human Counting",
    "Object Counting",
    "Fine-grained Action and Pose",
    "Human-Object Interaction",
    "Human Existence",
    "Object Existence",
    "Relative Distance to Human",
    "Relative Distance to Object",
    "View-Specific Depth Ordering",
    "View-Specific Direction",
]


# ---------------------------------------------------------------------------
# API keys are read from the environment only. Never hard-code them.
# ---------------------------------------------------------------------------
def get_gemini_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_openai_key():
    return os.environ.get("OPENAI_API_KEY")
