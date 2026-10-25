"""
Active setting: multi-turn exocentric view exploration.

The model starts from a single (occluded) view. At every turn it either answers (A-E) or
moves to the adjacent left (X, clockwise) / right (Y, counter-clockwise) viewpoint.
The episode ends when the model answers, outputs an invalid action, or reaches MAX_TURN views.

Examples:
    python evaluation/run_active.py --config configs/models/gemini-3.1-flash-lite.yaml
    python evaluation/run_active.py --config configs/models/qwen3-vl-8b-instruct.yaml
    python evaluation/run_active.py --config configs/models/qwen3-vl-8b-instruct.yaml --tensor_parallel_size 2
"""

import argparse

from tqdm import tqdm

from common import add_common_args, ensure_images, parse_args, resolve_results_dir
from sate_bench.config import ACTIVE_ANNOTATIONS, MAX_TURN
from sate_bench.data import load_samples
from sate_bench.metrics import ResultRecorder
from sate_bench.models import GenerationError, build_backend
from sate_bench.parsing import ANSWER_LETTERS, FallbackExtractor, parse_active_letter
from sate_bench.prompts import build_active_prompts


def run_episode(sample, backend, extractor, max_turns=MAX_TURN):
    """Runs one Active episode. Returns the list of per-turn rows, or None on an API failure."""
    system_prompt, user_prompt = build_active_prompts(sample)
    chat = backend.new_chat(system_prompt)
    image_paths = sample["image_paths"]
    num_views = len(image_paths)
    gt = str(sample["answer"]).strip().upper()
    view = int(sample["active_init_view"]) - 1

    rows = []
    for turn in range(1, max_turns + 1):
        row = {"index": sample["index"], "question_type": sample["type"], "question_name": sample["question_name"],
               "turn": turn, "view": view + 1, "raw_output": None, "thought_process": "", "action": None,
               "pred_answer": None, "gt_answer": gt, "correct": None}
        try:
            answer, thought = chat.send([image_paths[view]], user_prompt)
        except GenerationError as e:
            if backend.is_api:
                return None  # transient API failure: the sample is retried when the script is re-run
            row.update(raw_output=f"ERROR: {e}", pred_answer="ERROR", correct=0)
            rows.append(row)
            break

        action = parse_active_letter(answer) or extractor.extract_active(answer)
        row.update(raw_output=answer, thought_process=thought, action=action)
        rows.append(row)

        if action in ANSWER_LETTERS:
            row.update(pred_answer=action, correct=int(action == gt))
            break
        if action in ("X", "Y"):
            # X: adjacent left (clockwise) = next view; Y: adjacent right (counter-clockwise) = previous view
            view = (view + 1) % num_views if action == "X" else (view - 1) % num_views
            continue
        break  # invalid action ends the episode

    if rows[-1]["correct"] is None:  # invalid action or exploration budget exhausted
        rows[-1]["correct"] = 0
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--max_turns", type=int, default=MAX_TURN,
                        help=f"Maximum number of observed views per question (default: {MAX_TURN}).")
    args = parse_args(parser, "active")

    annotations = args.annotations or ACTIVE_ANNOTATIONS
    samples = load_samples(annotations, args.data_root, args.limit)
    ensure_images(samples)

    run_info = {"model": args.model, "backend": args.backend, "config": args.config, "max_turns": args.max_turns,
                "annotations": str(annotations)}
    recorder = ResultRecorder(resolve_results_dir(args, "active"), "active", run_info)
    todo = [s for s in samples if int(s["index"]) not in recorder.done]

    backend = build_backend(args, "active")
    extractor = FallbackExtractor()

    failed = []
    for sample in tqdm(todo, desc=f"Active [{args.model}]"):
        rows = run_episode(sample, backend, extractor, args.max_turns)
        if rows is None:
            failed.append(sample["index"])
            continue
        recorder.add(rows)

    metrics = recorder.write_metrics()
    print(f"\nResults saved to {recorder.results_dir}")
    print(f"Accuracy {metrics.get('accuracy', 0) * 100:.1f} | Steps {metrics.get('avg_steps', 0):.2f} | "
          f"Reward {metrics.get('avg_reward', 0) * 100:.1f} | Active rate {metrics.get('active_rate', 0) * 100:.1f}")
    if failed:
        print(f"{len(failed)} samples failed because of API errors; re-run the same command to resume them.")


if __name__ == "__main__":
    main()
