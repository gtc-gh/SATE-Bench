---
license: cc-by-nc-4.0
language:
  - en
pretty_name: SATE-Bench
size_categories:
  - 1K<n<10K
task_categories:
  - visual-question-answering
tags:
  - multi-view
  - spatial-reasoning
  - active-perception
  - human-centric
  - multimodal-llm
  - benchmark
configs:
  - config_name: active
    data_files:
      - split: test
        path: benchmark/spatial_active.json
  - config_name: passive
    data_files:
      - split: test
        path: benchmark/spatial_passive.json
  - config_name: grpo_training
    data_files:
      - split: train
        path: train/train.json
      - split: validation
        path: train/val.json
# The fields below only take effect if "Access requests" (gated access) is enabled in the dataset settings.
extra_gated_prompt: >-
  SATE-Bench is released for non-commercial research only. The images show real people and are derived
  from third-party datasets that keep their own licenses. By requesting access, you agree to use SATE-Bench
  only for non-commercial research, not to use it for person re-identification, surveillance or any attempt
  to identify individuals, and to respect the licenses of the source datasets.
extra_gated_fields:
  Name: text
  Affiliation: text
  I agree to use this dataset for non-commercial research only and to respect the licenses of the source datasets: checkbox
  I will not use this dataset for person re-identification or surveillance: checkbox
---

# SATE-Bench

**Humans in Space: Can MLLMs Construct Human-Centric Spatial Understanding via Active Exocentric Exploration?** (WACV 2027)

[🌐 Project Page](https://gtc-gh.github.io/SATE-Bench/) · [💻 Code](https://github.com/gtc-gh/SATE-Bench) · 📄 Paper (coming soon)

![SATE-Bench: Active exploration vs. Passive observation](https://gtc-gh.github.io/SATE-Bench/static/images/teaser.webp)

SATE-Bench is an interactive benchmark that evaluates the **visual self-awareness** of multimodal large language models (MLLMs): whether a model recognizes that its current view is insufficient and actively explores other exocentric viewpoints until it has enough evidence to answer.

- **Active setting** (1,615 questions): the model starts from a single, heavily occluded view and, at every turn, either answers or moves to the adjacent left (clockwise) or right (counter-clockwise) viewpoint. Exploration efficiency is scored with a step-penalized reward.
- **Passive setting** (910 questions): the model receives a subset of views that is guaranteed to contain all the evidence, which isolates multi-view fusion from exploration.
- **GRPO training data** (19,370 single-view samples): used in the paper to study whether active view-seeking can be learned with reinforcement learning.

This repository contains the annotations and the multi-view frames. The evaluation and training code is on [GitHub](https://github.com/gtc-gh/SATE-Bench).

## Dataset at a Glance

| | Benchmark | GRPO training data |
|---|---|---|
| Scenes | 316 (10 source datasets) | 86 (M3GYM, Ego-Exo4D) |
| QA pairs | 2,525 (1,615 Active + 910 Passive) | 19,370 (17,433 train + 1,937 validation) |
| Views per scene | 4–8, ordered clockwise | single view per sample |
| Frames | 1,907 (1,746 included + 161 Ego-Exo4D via script) | 1,937 (1,229 included + 708 Ego-Exo4D via script) |
| Resolution | 1920×1080 JPEG | 1920×1080 JPEG |

The ten tasks cover five cognitive dimensions:

| Dimension | Task | # QA |
|---|---|---|
| Entity Counting | Human Counting / Object Counting | 308 / 232 |
| Entity Existence | Human Existence / Object Existence | 203 / 218 |
| Pose and Physical Interaction | Fine-grained Action and Pose / Human-Object Interaction | 301 / 165 |
| Global Relative Proximity | Relative Distance to Human / Relative Distance to Object | 508 / 334 |
| View-Conditioned Spatial Geometry | View-Specific Depth Ordering / View-Specific Direction | 124 / 132 |

Most questions have 5 options, one of which is an abstention option (the target is not visible or the views are insufficient); existence questions have 2 options (Yes/No).

## Dataset Structure

```
benchmark/
├── spatial_active.json          # 1,615 Active-setting questions
├── spatial_passive.json         # 910 Passive-setting questions
└── images/<Dataset>/<sequence>/<frame>/view_<k>.jpg
train/
├── train.json                   # 17,433 GRPO training samples
├── val.json                     # 1,937 GRPO validation samples
└── images/M3GYM/{explore,answer}/<sequence>_view<k>_<i>.jpg
```

Ego-Exo4D frames (`benchmark/images/Ego-Exo4D/...` and `train/images/Ego-Exo4D/...`) are **not included** because the Ego-Exo4D license does not allow redistribution. They are referenced by the annotations and can be rebuilt exactly with the script in the code repository (see [How to Use](#how-to-use)).

### Benchmark fields

Example (`benchmark/spatial_active.json`):

```json
{
  "index": 0,
  "dataset": "Ego-Exo4D",
  "scene": "cmu_soccer06_5/0",
  "type": "active",
  "question_name": "Human Counting",
  "question": "Based on all the provided views, how many unique people can be identified in the entire indoor training facility?",
  "options": {"A": "6", "B": "9", "C": "The queried entity does not exist or is not visible, or the provided images are insufficient to answer the question.", "D": "8", "E": "7"},
  "answer": "E",
  "num_views": 4,
  "image_paths": ["images/Ego-Exo4D/cmu_soccer06_5/0/view_1.jpg", "...", "images/Ego-Exo4D/cmu_soccer06_5/0/view_4.jpg"],
  "active_init_view": 4
}
```

| Field | Type | Description |
|---|---|---|
| `index` | int | Unique identifier of the question within its file. |
| `dataset` | str | Source dataset of the scene. |
| `scene` | str | `<sequence>/<frame>` identifier of the synchronized multi-view frame. |
| `type` | str | `active` or `passive`. |
| `question_name` | str | Task name (one of the ten tasks above). |
| `question` | str | Natural-language question. |
| `options` | dict | Answer options `A`–`E` (or `A`–`B`). |
| `answer` | str | Ground-truth option letter. |
| `num_views` | int | Number of camera views of the scene (4–8). |
| `image_paths` | list[str] | All views of the scene, relative to `benchmark/`, ordered clockwise (`view_1`, `view_2`, ...). Moving "left" goes from `view_k` to `view_{k+1}`. |
| `active_init_view` | int | *Active only.* 1-based index of the initial, occluded view. |
| `passive_views` | list[int] | *Passive only.* 1-based indices of the views given to the model. |

### GRPO training fields

`train/train.json` and `train/val.json` contain single-view samples with the fields `index`, `dataset`, `type`, `question_name`, `question`, `options`, `answer` and `image_path` (relative to `train/`). For `active` samples, the correct option is *"Move and explore more views."*; for `passive` samples it is the factual answer.

## How to Use

The full evaluation protocol (multi-turn exploration, answer parsing and metrics) is implemented in the [code repository](https://github.com/gtc-gh/SATE-Bench):

```bash
git clone https://github.com/gtc-gh/SATE-Bench.git
cd SATE-Bench
pip install -r requirements.txt

# 1. Download this dataset into ./data
#    (if access requests are enabled, log in first with `hf auth login`)
python scripts/download_data.py

# 2. Rebuild the Ego-Exo4D frames from the official Ego-Exo4D videos (requires the Ego-Exo4D license)
python data_preparation/egoexo4d/extract_frames.py --egoexo_root <EGOEXO_DIR>/takes

# 3. Evaluate a model
python evaluation/run_active.py  --config configs/models/qwen3-vl-8b-instruct.yaml
python evaluation/run_passive.py --config configs/models/qwen3-vl-8b-instruct.yaml
```

The annotations can also be loaded with 🤗 Datasets (the configs are `active`, `passive` and `grpo_training`):

```python
from datasets import load_dataset

active = load_dataset("<HF_ORG>/SATE-Bench", "active", split="test")
print(active[0]["question"], active[0]["image_paths"])
```

## Source Datasets

SATE-Bench is built on frames from the official test splits (or held-out subsets) of ten human-centric multi-view datasets. The GRPO training data comes from disjoint training sequences of M3GYM and Ego-Exo4D.

| Source dataset | Scenes | Benchmark frames | Included in this repository | License of the source dataset |
|---|---|---|---|---|
| [4D-OR](https://github.com/egeozsoy/4D-OR) | 10 | 60 | Frames | MIT |
| [core4D](https://github.com/leolyliu/CORE4D-Instructions) | 96 | 384 | Frames | CC BY 4.0 |
| [Ego-Exo4D](https://ego-exo4d-data.org/) | 38 | 161 | Extraction script only | Ego4D License Agreement |
| [EgoHumans](https://rawalkhirodkar.github.io/egohumans/) | 10 | 80 | Frames | Apache-2.0 |
| [Harmony4D](https://jyuntins.github.io/harmony4d/) | 39 | 312 | Frames | MIT |
| [HOI-M3](https://juzezhang.github.io/HOIM3_ProjectPage/) | 21 | 168 | Frames | Apache-2.0 |
| [M3GYM](https://finalyou.github.io/M3GYM/) | 14 | 112 | Frames | Non-commercial academic |
| [MM-OR](https://github.com/egeozsoy/MM-OR) | 18 | 90 | Frames | Apache-2.0 |
| [MvMHAT](https://github.com/realgump/MvMHAT) | 5 | 20 | Frames | Non-commercial academic |
| [Replay](https://github.com/facebookresearch/replay_dataset) | 65 | 520 | Frames | CC BY-NC 4.0 |

Frames were synchronized with the official calibration data, extracted at 1920×1080 and sorted clockwise around each scene. Please also comply with the license and terms of use of each source dataset.

## Annotation Process

Four trained annotators wrote challenging multiple-choice questions on synchronized frames. For the Active setting they selected an initial view that genuinely requires exploration; for the Passive setting they selected a subset of views that guarantees the question is answerable. Two independent reviewers cross-validated every item (with a third expert resolving disputes); ambiguous items and items answerable from text alone were removed.

## Considerations for Using the Data

- **Personal information.** The frames show real people recorded by the source datasets, which obtained the consent of their participants; some source datasets blur faces. SATE-Bench adds no identity labels, and annotations describe people only by appearance (e.g. clothing). Do not use the data to identify individuals.
- **Intended use.** Evaluating and studying multi-view spatial reasoning and active view exploration of MLLMs, and research on learning to explore (e.g. the provided GRPO training data). The benchmark questions are meant for evaluation and should not be used for training.
- **Out-of-scope use.** Commercial use, person re-identification, surveillance, or any use that violates the terms of the source datasets.
- **Limitations.** Exploration is restricted to the discrete, pre-recorded camera views of each scene (4–8 per scene); all questions are in English; scenes cover indoor rooms, sports and gym settings, operating rooms and outdoor activities.

## License

The SATE-Bench annotations and data are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only. Frames derived from the source datasets additionally remain subject to the licenses and terms of use of those datasets. The code on GitHub is released under the Apache License 2.0.

## Citation

```bibtex
@inproceedings{satebench2027,
  title     = {Humans in Space: Can MLLMs Construct Human-Centric Spatial Understanding via Active Exocentric Exploration?},
  author    = {Guo, Tianchen and Zhang, Tengyue and Liu, Chen and Chen, Ling and Yu, Xin},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
