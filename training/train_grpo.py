"""
GRPO fine-tuning for single-turn Active/Passive decision making (Sec. 5 of the paper).

Each training sample is a single view + a 5-option question. For Active samples the correct
option is "Move and explore more views."; for Passive samples it is the factual answer.
The model learns to reason inside <think>...</think> and answer with \\boxed{LETTER}.

Usage:
    bash training/train.sh                                  # settings from training/config.yaml
    bash training/train.sh --action_weight 0.6 --output_dir checkpoints/grpo_action0.6
"""

import argparse
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from datasets import Dataset, Image as HFImage, Sequence
from peft import LoraConfig
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback
from trl import GRPOConfig, GRPOTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from prompts import build_prompt  # noqa: E402
from rewards import combined_reward_func  # noqa: E402

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def patch_trl_vllm_weight_sync():
    """Skips syncing vision-tower weights to the colocated vLLM engine.

    With TRL 1.x + vLLM, pushing the vision tower parameters fails because of parameter naming
    differences. Only the language model is trained with LoRA, so these weights never change.
    """
    try:
        import trl.generation.vllm_generation as vllm_generation
    except ImportError:
        return
    original_push = vllm_generation.VLLMGeneration._push_param_to_vllm

    def _patched_push(self, name, param):
        if "vision_tower" in name:
            return
        original_push(self, name, param)

    vllm_generation.VLLMGeneration._push_param_to_vllm = _patched_push


def resolve_path(p) -> Path:
    """Resolves a path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT_DIR / p


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_vqa_dataset(json_path: Path, logger: logging.Logger, image_size: tuple):
    """Loads train.json / val.json. Image paths are relative to the folder of the annotation file."""
    logger.info(f"Loading dataset from {json_path} (images resized to {image_size[0]}x{image_size[1]})...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted, missing = [], 0
    for item in tqdm(data, desc=f"Loading {json_path.name}", leave=False):
        img_path = json_path.parent / item["image_path"]
        if not img_path.exists():
            missing += 1
            continue
        formatted.append({
            "prompt": build_prompt(item),
            "images": [str(img_path)],
            "answer": str(item.get("answer", "")),
            "type": str(item.get("type", "passive")),
            "options": item.get("options", {}),
            "question_name": str(item.get("question_name", "")),
        })

    if missing:
        logger.warning(f"Skipped {missing} samples with missing images "
                       f"(did you run data_preparation/egoexo4d/extract_frames.py?).")
    logger.info(f"Loaded {len(formatted):,} samples from {json_path.name}")

    dataset = Dataset.from_list(formatted)
    dataset = dataset.cast_column("images", Sequence(HFImage()))

    def resize_on_the_fly(examples):
        examples["images"] = [[img.convert("RGB").resize(image_size, Image.BILINEAR) for img in img_list]
                              for img_list in examples["images"]]
        return examples

    dataset.set_transform(resize_on_the_fly)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for SATE-Bench")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default=None, help="'wandb', 'tensorboard' or 'none'")
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    # Optimisation
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    # GRPO
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--num_generations_eval", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None, help="KL penalty coefficient")
    # Rewards
    parser.add_argument("--accuracy_weight", type=float, default=None)
    parser.add_argument("--format_weight", type=float, default=None)
    parser.add_argument("--action_weight", type=float, default=None)
    # LoRA
    parser.add_argument("--no_lora", action="store_true", help="Full fine-tuning instead of LoRA.")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma separated, e.g. q_proj,k_proj")
    # vLLM
    parser.add_argument("--no_vllm", action="store_true", help="Disable vLLM-accelerated rollouts.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--vllm_max_model_length", type=int, default=None)
    return parser.parse_args()


def merge_config_with_args(cfg: dict, args) -> dict:
    """YAML values are the defaults; command-line arguments override them."""
    def pick(cli_value, section, key, default=None):
        return cli_value if cli_value is not None else cfg.get(section, {}).get(key, default)

    target_modules = ([m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
                      if args.lora_target_modules else cfg.get("lora", {}).get("target_modules", ["q_proj", "k_proj"]))
    return {
        "model_name_or_path": pick(args.model_name_or_path, "model", "model_name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "train_data": pick(args.train_data, "data", "train_data", "data/train/train.json"),
        "val_data": pick(args.val_data, "data", "val_data", "data/train/val.json"),
        "output_dir": pick(args.output_dir, "output", "output_dir", "checkpoints/grpo"),
        "report_to": pick(args.report_to, "output", "report_to", "none"),
        "logging_steps": pick(args.logging_steps, "output", "logging_steps", 1),
        "save_steps": pick(args.save_steps, "output", "save_steps", 30),
        "eval_steps": pick(args.eval_steps, "output", "eval_steps", 30),
        "image_size": (pick(args.image_width, "image", "resize_width", 640),
                       pick(args.image_height, "image", "resize_height", 360)),
        "learning_rate": float(pick(args.learning_rate, "training", "learning_rate", 1e-5)),
        "max_steps": pick(args.max_steps, "training", "max_steps", 300),
        "warmup_steps": pick(args.warmup_steps, "training", "warmup_steps", 30),
        "per_device_train_batch_size": pick(args.per_device_train_batch_size, "training", "per_device_train_batch_size", 2),
        "per_device_eval_batch_size": pick(args.per_device_eval_batch_size, "training", "per_device_eval_batch_size", 16),
        "gradient_accumulation_steps": pick(args.gradient_accumulation_steps, "training", "gradient_accumulation_steps", 16),
        "early_stopping_patience": cfg.get("training", {}).get("early_stopping_patience", 3),
        "num_generations": pick(args.num_generations, "grpo", "num_generations", 8),
        "num_generations_eval": pick(args.num_generations_eval, "grpo", "num_generations_eval", None),
        "max_completion_length": pick(args.max_completion_length, "grpo", "max_completion_length", 2048),
        "temperature": float(pick(args.temperature, "grpo", "temperature", 1.0)),
        "beta": float(pick(args.beta, "grpo", "beta", 0.005)),
        "accuracy_weight": float(pick(args.accuracy_weight, "rewards", "accuracy_weight", 1.0)),
        "format_weight": float(pick(args.format_weight, "rewards", "format_weight", 0.5)),
        "action_weight": float(pick(args.action_weight, "rewards", "action_weight", 0.2)),
        "use_lora": False if args.no_lora else cfg.get("lora", {}).get("use_lora", True),
        "lora_r": pick(args.lora_r, "lora", "r", 16),
        "lora_alpha": pick(args.lora_alpha, "lora", "lora_alpha", 16),
        "lora_dropout": float(pick(args.lora_dropout, "lora", "lora_dropout", 0.1)),
        "target_modules": target_modules,
        "use_vllm": False if args.no_vllm else cfg.get("vllm", {}).get("use_vllm", True),
        "vllm_gpu_memory_utilization": float(pick(args.vllm_gpu_memory_utilization, "vllm", "vllm_gpu_memory_utilization", 0.4)),
        "vllm_max_model_length": int(pick(args.vllm_max_model_length, "vllm", "vllm_max_model_length", 8192)),
    }


def main():
    args = parse_args()
    cfg = merge_config_with_args(load_yaml_config(resolve_path(args.config)), args)

    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(str(log_dir / "train_grpo.log")), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info("SATE-Bench GRPO training configuration:\n" + json.dumps(cfg, indent=2, default=str))

    if cfg["use_vllm"]:
        patch_trl_vllm_weight_sync()

    # 1. Data
    train_dataset = load_vqa_dataset(resolve_path(cfg["train_data"]), logger, cfg["image_size"])
    val_dataset = load_vqa_dataset(resolve_path(cfg["val_data"]), logger, cfg["image_size"]) \
        if cfg.get("val_data") else None

    # 2. Model and processor
    processor = AutoProcessor.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))
        device_map = None
    else:
        device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_name_or_path"], dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True)

    peft_config = None
    if cfg["use_lora"]:
        peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                                 target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")

    # 3. GRPO config. The global eval batch size must be divisible by the number of generations.
    eval_bs = cfg["per_device_eval_batch_size"]
    divisor = cfg["num_generations_eval"] or cfg["num_generations"]
    if val_dataset is not None and eval_bs % divisor != 0:
        eval_bs = ((eval_bs + divisor - 1) // divisor) * divisor
        logger.info(f"per_device_eval_batch_size set to {eval_bs} (divisible by {divisor})")

    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        max_steps=cfg["max_steps"],
        warmup_steps=cfg["warmup_steps"],
        save_steps=cfg["save_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=cfg["eval_steps"] if val_dataset is not None else None,
        bf16=True,
        remove_unused_columns=False,  # keeps the 'images' column for multimodal inputs
        dataloader_drop_last=True,
        num_generations=cfg["num_generations"],
        num_generations_eval=cfg["num_generations_eval"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        beta=cfg["beta"],
        use_vllm=cfg["use_vllm"],
        vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        vllm_max_model_length=cfg["vllm_max_model_length"],
        report_to=cfg["report_to"],
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_reward" if val_dataset is not None else None,
        greater_is_better=True,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])] \
        if val_dataset is not None else []

    reward_fn = partial(combined_reward_func, accuracy_weight=cfg["accuracy_weight"],
                        format_weight=cfg["format_weight"], action_weight=cfg["action_weight"])
    reward_fn.__name__ = "combined_reward"

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.train()

    logger.info(f"Saving model and processor to {cfg['output_dir']}")
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
