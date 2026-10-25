"""
Downloads the SATE-Bench images from the Hugging Face Hub into ./data.

    python scripts/download_data.py                    # benchmark + training images
    python scripts/download_data.py --subset benchmark # evaluation images only

Ego-Exo4D frames are not redistributed; extract them afterwards with
    python data_preparation/egoexo4d/extract_frames.py --egoexo_root <EGO_EXO4D_TAKES_DIR>
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

# TODO: replace with the final Hugging Face dataset repository before release.
HF_DATASET_REPO = "<HF_ORG>/SATE-Bench"
REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default=HF_DATASET_REPO)
    parser.add_argument("--subset", choices=["all", "benchmark", "train"], default="all")
    parser.add_argument("--local_dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    patterns = {"benchmark": ["benchmark/**"], "train": ["train/**"]}
    allow = patterns["benchmark"] + patterns["train"] if args.subset == "all" else patterns[args.subset]
    snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=args.local_dir, allow_patterns=allow)
    print(f"Downloaded {args.subset} data to {args.local_dir}")


if __name__ == "__main__":
    main()
