"""
Builds data/benchmark/metadata.jsonl, which lets the Hugging Face Dataset Viewer and Data Studio show
every benchmark question together with its multi-view images (config `browse` in data/README.md).

Ego-Exo4D questions are skipped because their frames are not redistributed; they remain available in
the `active` and `passive` annotation files. Re-run this script whenever the annotations change.

    python scripts/build_viewer_metadata.py
"""

import json
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent.parent / "data" / "benchmark"


def main():
    rows = []
    for setting in ("active", "passive"):
        for s in json.loads((BENCHMARK / f"spatial_{setting}.json").read_text()):
            if s["dataset"] == "Ego-Exo4D":
                continue
            rows.append({
                "file_names": s["image_paths"],           # all views, clockwise; shown as an image list
                "setting": setting,
                "index": s["index"],
                "question_name": s["question_name"],
                "dataset": s["dataset"],
                "scene": s["scene"],
                "question": s["question"],
                "options": s["options"],
                "answer": s["answer"],
                "num_views": s["num_views"],
                "active_init_view": s.get("active_init_view"),
                "passive_views": s.get("passive_views"),
            })
    with open(BENCHMARK / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions to {BENCHMARK / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
