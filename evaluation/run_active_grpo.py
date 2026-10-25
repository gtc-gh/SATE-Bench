"""
Active-setting evaluation of GRPO fine-tuned models (Sec. 5 of the paper).

The tuned model keeps the reasoning format used during training (<think>...</think> followed by
\boxed{LETTER}). The exploration protocol is identical to evaluation/run_active.py, except that
exploration is disabled on the last allowed turn and the model is asked for a final answer.

Example:
    python evaluation/run_active_grpo.py --lora_path checkpoints/grpo_qwen25vl_3b_action0.2
    # Passive setting of the same checkpoint:
    python evaluation/run_passive.py --backend vllm --model Qwen/Qwen2.5-VL-3B-Instruct \
        --lora_path checkpoints/grpo_qwen25vl_3b_action0.2 --enable_thinking true --max_tokens 4096
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from common import ensure_images
from sate_bench.config import ACTIVE_ANNOTATIONS, MAX_TURN, RESULTS_ROOT
from sate_bench.data import load_image, load_samples
from sate_bench.metrics import ResultRecorder
from sate_bench.models import VLLMBackend
from sate_bench.parsing import FallbackExtractor, extract_boxed_choice, parse_active_letter, split_thinking

MAX_GRACE_TRIALS = 3


# ==============================================================================
# GRPO Combined System Prompt for Active Setting
# Combines GRPO reasoning & \boxed{} format with 3D multi-view navigation (X/Y)
# ==============================================================================
GRPO_SYSTEM_PROMPT_ACTIVE = """You are an expert in visual reasoning and 3D spatial understanding. \
You are currently viewing 1 out of {num_views} possible camera viewpoints of a single scene. \
These viewpoints are uniformly distributed in a circle around the scene and sorted in clockwise order.

Your task is to carefully analyze the visual evidence, answer the question, and make a decision:
1. ANSWER: If the visual evidence in the current view is clear, unambiguous, and sufficient to confidently answer the question, choose the correct factual option from Options.
2. EXPLORE: If the visual evidence in the current view is occluded, insufficient, or requires observing the scene from other angles, choose an exploration option from Exploration Options:
   - Option 'X': Move to adjacent left (clockwise) viewpoint.
   - Option 'Y': Move to adjacent right (counter-clockwise) viewpoint.

OUTPUT FORMAT REQUIREMENT:
You must first write your step-by-step reasoning between <think> and </think> tags.
Then, provide your final chosen option letter formatted strictly within \\boxed{LETTER} (for example, \\boxed{A} or \\boxed{X}).

IMPORTANT NOTES:
Do not always explore; only explore when necessary. Your goal is to answer the question correctly with minimal exploration turns. You should stop the exploration and answer the question as soon as you have sufficient evidence.
"""


# ==============================================================================
# Final Turn System Prompt (Strictly disables exploration)
# ==============================================================================
GRPO_SYSTEM_PROMPT_ACTIVE_FINAL = """You are an expert in visual reasoning and 3D spatial understanding. \
You have completed all allowed exploration moves and have reached the FINAL DECISION TURN.

STRICT FINAL TURN RULES:
1. EXPLORATION IS PERMANENTLY DISABLED. Moving actions 'X' and 'Y' DO NOT EXIST and are STRICTLY FORBIDDEN.
2. You MUST make your final definitive choice exclusively from the factual multiple-choice options (Options).

OUTPUT FORMAT REQUIREMENT:
You must first write your step-by-step reasoning between <think> and </think> tags.
Then, provide your final chosen option letter formatted strictly within \\boxed{LETTER} (for example, \\boxed{A} or \\boxed{B})."""


def format_options_text(options: dict) -> str:
    """Formats the options dictionary into lines like 'A) ... \nB) ...'."""
    keys = sorted(options.keys())
    return "\n".join([f"{k}) {options[k]}" for k in keys])


def build_grpo_active_user_prompt(question: str, options: dict) -> str:
    """
    Constructs the combined user prompt for the active setting:
    - Retains GRPO training style with thinking process and in-context reasoning examples
    - Formats the 5 (or 2) options directly loaded from benchmark annotations
    - Provides the 2 standard exploration actions: X (move left) and Y (move right)
    """
    options_text = format_options_text(options)
    opt_keys = sorted(options.keys())
    allowed_list = opt_keys + ["X", "Y"]
    allowed_str = ", ".join(allowed_list)

    user_text = (
        "Please carefully analyze the provided image and answer the question. "
        "Do not always explore; only explore when necessary. Your goal is to answer the question correctly with minimal exploration turns. You should stop the exploration and answer the question as soon as you have sufficient evidence. "
        "You MUST format your output exactly as shown in the examples below.\n\n"
        "--- Example 1 ---\n"
        "Question:\n"
        "Is the person in the black jacket holding a red cup?\n\n"
        "Options:\n"
        "A) Yes\n"
        "B) No\n\n"
        "Exploration Options:\n"
        "X) Move to adjacent left (clockwise) viewpoint\n"
        "Y) Move to adjacent right (counter-clockwise) viewpoint\n\n"
        "Output:\n"
        "<think>\n"
        "In the current view, the person in the black jacket is clearly visible in the foreground. Their right hand is resting on the table holding a bright red ceramic cup. The visual evidence is clear and unambiguous to answer the question directly.\n"
        "</think>\n"
        "\\boxed{A}\n"
        "--- End of Example 1 ---\n\n"
        "Now, answer the following question:\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Exploration Options:\n"
        "X) Move to adjacent left (clockwise) viewpoint\n"
        "Y) Move to adjacent right (counter-clockwise) viewpoint\n\n"
        f"Output:\n"
        f"First output your thinking process inside <think>...</think>, and then output your final answer as \\boxed{{LETTER}} choosing from [{allowed_str}]."
    )
    return user_text


def build_grpo_active_final_turn_prompt(question: str, options: dict) -> str:
    """
    Constructs the forced-answer user prompt for the final active turn:
    - Removes all exploration options (no X, no Y).
    - Explicitly forbids moving/exploring and forces selecting from factual options.
    """
    options_text = format_options_text(options)
    opt_keys = sorted(options.keys())
    allowed_str = ", ".join(opt_keys)

    user_text = (
        "[FINAL TURN - EXPLORATION STRICTLY DISABLED]\n"
        "You have reached the final viewpoint and maximum turn limit. "
        "Exploration actions X and Y are permanently disabled and invalid. "
        "You MUST choose one factual option from the list below.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Output Requirement:\n"
        f"First output your thinking process inside <think>...</think>, and then output your final answer strictly as \\boxed{{LETTER}} choosing ONLY from [{allowed_str}]. Do NOT output X or Y."
    )
    return user_text



def extract_active_action(raw_text: str, answer_text: str, valid_keys: list):
    """\\boxed{} extraction first, then rule-based letter parsing. Returns None if nothing valid is found."""
    for text in (raw_text, answer_text):
        choice = extract_boxed_choice(text, valid_keys)
        if choice:
            return choice
    parsed = parse_active_letter(answer_text)
    return parsed if parsed in valid_keys else None


class GRPOActiveEvaluator:
    def __init__(self, backend: VLLMBackend, extractor: FallbackExtractor):
        self.backend = backend
        self.extractor = extractor

    def generate(self, messages):
        return self.backend.run([self.backend.build_request(messages)])[0]

    def run_episode(self, sample):
        options = sample["options"]
        option_keys = sorted(options.keys())
        gt = str(sample["answer"]).strip().upper()
        image_paths = sample["image_paths"]
        num_views = len(image_paths)
        cur_view = int(sample["active_init_view"]) - 1

        messages = [{"role": "system", "content": GRPO_SYSTEM_PROMPT_ACTIVE.replace("{num_views}", str(num_views))}]
        rows = []
        for turn in range(1, MAX_TURN + 1):
            final_turn = turn == MAX_TURN
            if final_turn:
                messages[0]["content"] = GRPO_SYSTEM_PROMPT_ACTIVE_FINAL
                user_prompt = build_grpo_active_final_turn_prompt(sample["question"], options)
                valid_keys = option_keys
            else:
                user_prompt = build_grpo_active_user_prompt(sample["question"], options)
                valid_keys = option_keys + ["X", "Y"]

            image = load_image(image_paths[cur_view])
            messages.append({"role": "user", "content": [{"type": "image", "image": image},
                                                          {"type": "text", "text": user_prompt}]})
            row = {"index": sample["index"], "question_type": sample["type"], "question_name": sample["question_name"],
                   "turn": turn, "view": cur_view + 1, "raw_output": None, "thought_process": "", "action": None,
                   "pred_answer": None, "gt_answer": gt, "correct": None}
            try:
                raw = self.generate(messages)
            except Exception as e:  # noqa: BLE001
                row.update(raw_output=f"ERROR: {e}", pred_answer="ERROR", correct=0)
                rows.append(row)
                break

            answer_text, thought = split_thinking(raw)
            messages.append({"role": "assistant", "content": raw})
            action = extract_active_action(raw, answer_text, valid_keys)

            # On the final turn, re-prompt (at most MAX_GRACE_TRIALS times) until a factual option is chosen.
            if final_turn and action not in option_keys:
                allowed = ", ".join(option_keys)
                for trial in range(1, MAX_GRACE_TRIALS + 1):
                    correction = (
                        f"Moving/exploration is strictly disabled on the final turn (Option '{action}' is invalid). "
                        f"You MUST select one of the valid factual options: [{allowed}]. "
                        f"Please provide your reasoning in <think>...</think> and output your final answer strictly "
                        f"as \\boxed{{LETTER}} choosing from [{allowed}]."
                    )
                    try:
                        corr_raw = self.generate(messages + [{"role": "user",
                                                              "content": [{"type": "text", "text": correction}]}])
                    except Exception:  # noqa: BLE001
                        continue
                    corr_answer, corr_thought = split_thinking(corr_raw)
                    raw = f"{raw}\n[Final Turn Grace Trial {trial}]: {corr_raw}"
                    answer_text, thought = corr_answer, corr_thought or thought
                    action = extract_active_action(corr_raw, corr_answer, option_keys)
                    if action in option_keys:
                        break

            if action not in valid_keys:
                action = self.extractor.extract_active(answer_text)
                if final_turn and action not in option_keys:
                    action = None

            row.update(raw_output=raw.strip(), thought_process=thought, action=action)
            rows.append(row)
            if not final_turn and action in ("X", "Y"):
                cur_view = (cur_view + 1) % num_views if action == "X" else (cur_view - 1) % num_views
                continue
            if action in option_keys:
                row.update(pred_answer=action, correct=int(action == gt))
            else:
                row.update(pred_answer=action, correct=0)
            break
        return rows


def infer_base_model(lora_path: Path) -> str:
    cfg = lora_path / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text()).get("base_model_name_or_path") or "Qwen/Qwen2.5-VL-3B-Instruct"
    return "Qwen/Qwen2.5-VL-3B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora_path", required=True, help="GRPO checkpoint folder (LoRA adapter).")
    parser.add_argument("--model", default=None, help="Base model (default: read from adapter_config.json).")
    parser.add_argument("--annotations", default=str(ACTIVE_ANNOTATIONS))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    args = parser.parse_args()

    lora_path = Path(args.lora_path)
    model = args.model or infer_base_model(lora_path)
    samples = load_samples(args.annotations, args.data_root, args.limit)
    ensure_images(samples)

    results_dir = args.results_dir or RESULTS_ROOT / "active" / "grpo" / lora_path.resolve().name
    recorder = ResultRecorder(results_dir, "active", {"model": model, "lora_path": str(lora_path),
                                                      "annotations": args.annotations})
    todo = [s for s in samples if int(s["index"]) not in recorder.done]

    backend = VLLMBackend(model=model, max_images=10, tensor_parallel_size=args.tensor_parallel_size,
                          gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                          temperature=args.temperature, max_tokens=args.max_tokens,
                          repetition_penalty=args.repetition_penalty, seed=args.seed, enable_thinking=True,
                          lora_path=str(lora_path), max_lora_rank=args.max_lora_rank)
    evaluator = GRPOActiveEvaluator(backend, FallbackExtractor())
    for sample in tqdm(todo, desc=f"Active GRPO [{lora_path.name}]"):
        recorder.add(evaluator.run_episode(sample))

    m = recorder.write_metrics()
    print(f"\nResults saved to {recorder.results_dir}")
    print(f"Accuracy {m.get('accuracy', 0) * 100:.1f} | Steps {m.get('avg_steps', 0):.2f} | "
          f"Reward {m.get('avg_reward', 0) * 100:.1f} | Active rate {m.get('active_rate', 0) * 100:.1f}")


if __name__ == "__main__":
    main()
