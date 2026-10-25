"""Command-line arguments and helpers shared by the evaluation scripts."""

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sate_bench.config import RESULTS_ROOT  # noqa: E402
from sate_bench.data import check_images  # noqa: E402


def str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {v}")


def add_common_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("general")
    g.add_argument("--config", default=None,
                   help="Model config file (e.g. configs/models/qwen3-vl-8b-instruct.yaml). "
                        "Command-line options override the values in the file.")
    g.add_argument("--backend", choices=["gemini", "openai", "vllm"], default=None)
    g.add_argument("--model", default=None,
                   help="API model name (e.g. gemini-3.1-flash-lite, gpt-5-mini-2025-08-07) or HF model id for vLLM.")
    g.add_argument("--annotations", default=None, help="Annotation file (defaults to data/benchmark/spatial_<setting>.json).")
    g.add_argument("--data_root", default=None, help="Root of the relative image paths (defaults to the annotation folder).")
    g.add_argument("--results_dir", default=None, help="Output folder (defaults to results/<setting>/<run_name>).")
    g.add_argument("--run_name", default=None, help="Name of the output folder (defaults to the model name).")
    g.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples (for debugging).")
    g.add_argument("--openai_base_url", default=None, help="Optional OpenAI-compatible endpoint.")

    v = parser.add_argument_group("vLLM (open-source models)")
    v.add_argument("--tensor_parallel_size", type=int, default=1)
    v.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    v.add_argument("--max_model_len", type=int, default=16384)
    v.add_argument("--max_num_seqs", type=int, default=None)
    v.add_argument("--trust_remote_code", action="store_true")
    v.add_argument("--temperature", type=float, default=0.0)
    v.add_argument("--max_tokens", type=int, default=8192)
    v.add_argument("--repetition_penalty", type=float, default=1.05)
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--enable_thinking", type=str2bool, default=None,
                   help="Passed to the chat template (Qwen3 / Qwen3.5 style models). Omit for models without it.")
    v.add_argument("--lora_path", default=None, help="Optional LoRA adapter (e.g. a GRPO checkpoint folder).")
    v.add_argument("--max_lora_rank", type=int, default=64)


def load_model_config(path, setting: str) -> dict:
    """Reads a model config. Keys under `active:` / `passive:` only apply to that setting."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    values = {k: v for k, v in cfg.items() if k not in ("active", "passive")}
    values.update(cfg.get(setting) or {})
    return values


def parse_args(parser: argparse.ArgumentParser, setting: str) -> argparse.Namespace:
    """Parses the command line on top of the defaults given by `--config`."""
    pre, _ = parser.parse_known_args()
    if pre.config:
        values = load_model_config(pre.config, setting)
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(values) - known)
        if unknown:
            parser.error(f"unknown keys in {pre.config}: {', '.join(unknown)}")
        parser.set_defaults(**values)
    args = parser.parse_args()
    if not args.backend or not args.model:
        parser.error("specify --config, or both --backend and --model")
    return args


def default_run_name(args, suffix: str = "") -> str:
    if not args.run_name and args.config:
        name = Path(args.config).stem
    else:
        name = args.run_name or re.sub(r"[^A-Za-z0-9._-]+", "_", args.model.split("/")[-1])
    if not args.run_name and getattr(args, "lora_path", None):
        name += "-" + Path(args.lora_path.rstrip("/")).name
    return name + suffix


def resolve_results_dir(args, setting: str, suffix: str = "") -> Path:
    if args.results_dir:
        return Path(args.results_dir)
    return RESULTS_ROOT / setting / default_run_name(args, suffix)


def ensure_images(samples):
    missing = check_images(samples)
    if missing:
        hint = ""
        if any("/Ego-Exo4D/" in p for p in missing):
            hint = ("\nEgo-Exo4D frames are not redistributed. Run:\n"
                    "  python data_preparation/egoexo4d/extract_frames.py --egoexo_root <EGO_EXO4D_TAKES_DIR>")
        raise SystemExit(f"{len(missing)} images are missing, e.g. {missing[0]}{hint}")
