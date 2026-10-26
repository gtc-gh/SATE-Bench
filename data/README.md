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
  - config_name: browse
    default: true
    data_files:
      - split: test
        path:
          - benchmark/metadata.jsonl
          - benchmark/images/**
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

Most questions have 5 options, one of which is an abstention option (the target is not visible or the views are insufficient); existence questions have 2 options (Yes/No).

## How to Use

The evaluation and training code is in the [code repository](https://github.com/gtc-gh/SATE-Bench). Setting up the data takes two steps: download everything hosted here, then rebuild the Ego-Exo4D frames.

### Step 1: Download the data

This dataset is gated: accept the terms on this page and log in with `hf auth login` before downloading.

```bash
git clone https://github.com/gtc-gh/SATE-Bench.git
cd SATE-Bench
pip install -r requirements.txt

hf auth login
python scripts/download_data.py   # annotations and all frames except Ego-Exo4D, into ./data
```

### Step 2: Rebuild the Ego-Exo4D frames

The Ego-Exo4D license does not allow redistributing its frames, so the 161 benchmark frames (38 takes) and 708 training frames (55 takes) that come from Ego-Exo4D have to be rebuilt from the official videos. Only the 448p videos of these 93 takes are needed, not the full dataset (the default Ego-Exo4D download is about 14 TiB).

1. **Request access.** Review and sign the Ego-Exo4D license agreement at <https://ego4d.dev/request/ego-exo4d>. Approval usually takes about 48 hours; you then receive AWS access credentials by email. The credentials expire after 14 days, so download the videos within that window (request again at the same link to renew them).
2. **Install the downloader and set up the credentials:**
   ```bash
   pip install ego4d awscli
   aws configure   # enter the access key and secret key from the email; press Enter for the other fields
   ```
3. **Download the Ego-Exo4D metadata** (small JSON files, including `takes.json`):
   ```bash
   egoexo -o <EGOEXO_DIR> --parts metadata
   ```
4. **Download only the videos SATE-Bench needs.** The downloader selects takes by `take_uid`; this helper looks up the 93 take names in `takes.json` and prints the exact command (add `--split benchmark` if you only need the evaluation frames):
   ```bash
   python data_preparation/egoexo4d/download_command.py --takes_json <EGOEXO_DIR>/takes.json
   # then run the printed command, which looks like:
   # egoexo -o <EGOEXO_DIR> --parts downscaled_takes/448 --uids <take_uid_1> <take_uid_2> ...
   ```
5. **Rebuild the frames.** The script reads the exact frame index of every frame from `data_preparation/egoexo4d/*_frames.json` and writes it to the path referenced by the annotations:
   ```bash
   python data_preparation/egoexo4d/extract_frames.py --egoexo_root <EGOEXO_DIR>
   # evaluation frames only:
   python data_preparation/egoexo4d/extract_frames.py --egoexo_root <EGOEXO_DIR> --split benchmark
   ```
   In our setup (OpenCV 5.0) the rebuilt frames are byte-identical to the ones used in the paper; other OpenCV/FFmpeg versions may introduce tiny pixel differences.

### Step 3: Load with 🤗 Datasets (optional)

The `browse` config can also be loaded with 🤗 Datasets (this downloads the benchmark images). It contains all benchmark questions except the Ego-Exo4D ones, so it does not need Step 2:

```python
from datasets import load_dataset

browse = load_dataset("gtc-hf/SATE-Bench", split="test")
row = browse[0]
print(row["question"], row["options"], row["answer"], len(row["images"]))
```

For the complete annotations (including Ego-Exo4D questions and the GRPO training data), use the JSON files downloaded in Step 1.

### Step 4: Evaluate a model

```bash
python evaluation/run_active.py  --config configs/models/qwen3-vl-8b-instruct.yaml
python evaluation/run_passive.py --config configs/models/qwen3-vl-8b-instruct.yaml
```

See the [code repository](https://github.com/gtc-gh/SATE-Bench) for all model configs, the metrics and the GRPO training code.

## Dataset Structure

```
benchmark/
├── spatial_active.json          # 1,615 Active-setting questions
├── spatial_passive.json         # 910 Passive-setting questions
├── metadata.jsonl               # questions + image lists for the Dataset Viewer (config `browse`)
└── images/<Dataset>/<sequence>/<frame>/view_<k>.jpg
train/
├── train.json                   # 17,433 GRPO training samples
├── val.json                     # 1,937 GRPO validation samples
└── images/M3GYM/{explore,answer}/<sequence>_view<k>_<i>.jpg
```

Ego-Exo4D frames (`benchmark/images/Ego-Exo4D/...` and `train/images/Ego-Exo4D/...`) are **not included**; they are referenced by the annotations and rebuilt in [How to Use, Step 2](#how-to-use).

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
| `question_name` | str | Task name, one of ten tasks (e.g. `Human Counting`, `View-Specific Direction`). |
| `question` | str | Natural-language question. |
| `options` | dict | Answer options `A`–`E` (or `A`–`B`). |
| `answer` | str | Ground-truth option letter. |
| `num_views` | int | Number of camera views of the scene (4–8). |
| `image_paths` | list[str] | All views of the scene, relative to `benchmark/`, ordered clockwise (`view_1`, `view_2`, ...). Moving "left" goes from `view_k` to `view_{k+1}`. |
| `active_init_view` | int | *Active only.* 1-based index of the initial, occluded view. |
| `passive_views` | list[int] | *Passive only.* 1-based indices of the views given to the model. |

### GRPO training fields

`train/train.json` and `train/val.json` contain single-view samples with the fields `index`, `dataset`, `type`, `question_name`, `question`, `options`, `answer` and `image_path` (relative to `train/`). For `active` samples, the correct option is *"Move and explore more views."*; for `passive` samples it is the factual answer.

## Source Datasets

SATE-Bench is built on frames from the official test splits (or held-out subsets) of ten human-centric multi-view datasets. The GRPO training data comes from disjoint training sequences of M3GYM and Ego-Exo4D.

| Source dataset | Included in this repository | License of the source dataset |
|---|---|---|
| [4D-OR](https://github.com/egeozsoy/4D-OR) | Frames | MIT |
| [core4D](https://github.com/leolyliu/CORE4D-Instructions) | Frames | CC BY 4.0 |
| [Ego-Exo4D](https://ego-exo4d-data.org/) | Extraction script only | Ego4D License Agreement |
| [EgoHumans](https://rawalkhirodkar.github.io/egohumans/) | Frames | Apache-2.0 |
| [Harmony4D](https://jyuntins.github.io/harmony4d/) | Frames | MIT |
| [HOI-M3](https://juzezhang.github.io/HOIM3_ProjectPage/) | Frames | Apache-2.0 |
| [M3GYM](https://finalyou.github.io/M3GYM/) | Frames | Non-commercial academic |
| [MM-OR](https://github.com/egeozsoy/MM-OR) | Frames | Apache-2.0 |
| [MvMHAT](https://github.com/realgump/MvMHAT) | Frames | Non-commercial academic |
| [Replay](https://github.com/facebookresearch/replay_dataset) | Frames | CC BY-NC 4.0 |

Frames were synchronized with the official calibration data, extracted at 1920×1080 and sorted clockwise around each scene. Please also comply with the license and terms of use of each source dataset.

## Considerations for Using the Data

- **Personal information.** The frames show real people recorded by the source datasets, which obtained the consent of their participants; some source datasets blur faces. SATE-Bench adds no identity labels, and annotations describe people only by appearance (e.g. clothing). Do not use the data to identify individuals.
- **Intended use.** Evaluating and studying multi-view spatial reasoning and active view exploration of MLLMs, and research on learning to explore (e.g. the provided GRPO training data). The benchmark questions are meant for evaluation and should not be used for training.
- **Out-of-scope use.** Commercial use, person re-identification, surveillance, or any use that violates the terms of the source datasets.
- **Limitations.** Exploration is restricted to the discrete, pre-recorded camera views of each scene (4–8 per scene); all questions are in English; scenes cover indoor rooms, sports and gym settings, operating rooms and outdoor activities.

## License

The SATE-Bench annotations and data are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only. Frames derived from the source datasets additionally remain subject to the licenses and terms of use of those datasets. The code on GitHub is released under the Apache License 2.0.

## Citation

Coming soon.
