"""
Passive setting: single-turn multi-view question answering.

The model receives the annotated subset of views ("passive_views") that is guaranteed to
contain all the evidence required to answer the question.

Examples:
    python evaluation/run_passive.py --config configs/models/gemini-3.1-flash-lite.yaml
    python evaluation/run_passive.py --config configs/models/qwen3-vl-8b-instruct.yaml
"""

import argparse

from tqdm import tqdm

from common import add_common_args, ensure_images, parse_args, resolve_results_dir
from sate_bench.config import PASSIVE_ANNOTATIONS
from sate_bench.data import load_samples
from sate_bench.metrics import ResultRecorder
from sate_bench.models import GenerationError, build_backend
from sate_bench.parsing import FallbackExtractor, parse_passive_letter
from sate_bench.prompts import SYSTEM_PROMPT_PASSIVE, build_passive_prompt


def passive_views(sample):
    paths = sample["image_paths"]
    return [paths[v - 1] for v in sample["passive_views"] if 0 <= v - 1 < len(paths)]


def make_row(sample, answer, thought, extractor):
    gt = str(sample["answer"]).strip().upper()
    if answer.startswith("ERROR:"):
        pred = "ERROR"
    else:
        pred = parse_passive_letter(answer) or extractor.extract_passive(answer)
    return {"index": sample["index"], "question_type": sample["type"], "question_name": sample["question_name"],
            "raw_output": answer, "thought_process": thought, "pred_answer": pred, "gt_answer": gt,
            "correct": int(pred == gt)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parse_args(parser, "passive")

    annotations = args.annotations or PASSIVE_ANNOTATIONS
    samples = load_samples(annotations, args.data_root, args.limit)
    ensure_images(samples)

    run_info = {"model": args.model, "backend": args.backend, "config": args.config, "annotations": str(annotations)}
    recorder = ResultRecorder(resolve_results_dir(args, "passive"), "passive", run_info)
    todo = [s for s in samples if int(s["index"]) not in recorder.done]

    backend = build_backend(args, "passive")
    extractor = FallbackExtractor()
    inputs = [(SYSTEM_PROMPT_PASSIVE, passive_views(s), build_passive_prompt(s)) for s in todo]

    failed = []
    if backend.is_api:
        for sample, (system_prompt, images, text) in tqdm(list(zip(todo, inputs)), desc=f"Passive [{args.model}]"):
            try:
                answer, thought = backend.generate(system_prompt, images, text)
            except GenerationError:
                failed.append(sample["index"])
                continue
            recorder.add([make_row(sample, answer, thought, extractor)])
    elif todo:
        outputs = backend.generate_batch(inputs)
        for sample, (answer, thought) in tqdm(list(zip(todo, outputs)), desc="Parsing"):
            recorder.add([make_row(sample, answer, thought, extractor)])

    metrics = recorder.write_metrics()
    print(f"\nResults saved to {recorder.results_dir}")
    print(f"Accuracy {metrics.get('accuracy', 0) * 100:.1f} on {metrics.get('total_samples', 0)} samples")
    if failed:
        print(f"{len(failed)} samples failed because of API errors; re-run the same command to resume them.")


if __name__ == "__main__":
    main()
