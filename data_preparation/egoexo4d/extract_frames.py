"""
Reconstruct the Ego-Exo4D frames used by SATE-Bench from the official Ego-Exo4D videos.

Ego-Exo4D frames cannot be redistributed, so SATE-Bench only ships the exact frame
locations (``benchmark_frames.json`` / ``train_frames.json``). This script reads the
official ``frame_aligned_videos/downscaled/448/camXX.mp4`` videos and writes every
required frame to the path referenced by the SATE-Bench annotations.

Usage:
    python data_preparation/egoexo4d/extract_frames.py --egoexo_root /path/to/egoexo4d/takes
    python data_preparation/egoexo4d/extract_frames.py --egoexo_root ... --split benchmark
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = {
    "benchmark": Path(__file__).resolve().parent / "benchmark_frames.json",
    "train": Path(__file__).resolve().parent / "train_frames.json",
}


def resolve_take_dir(egoexo_root: Path, take_name: str) -> Path:
    """Supports both `<root>/<take>` and the official `<root>/takes/<take>` layout."""
    for candidate in (egoexo_root / take_name, egoexo_root / "takes" / take_name):
        if candidate.is_dir():
            return candidate
    return egoexo_root / take_name


def extract_split(manifest_path: Path, egoexo_root: Path, overwrite: bool) -> tuple:
    manifest = json.loads(manifest_path.read_text())
    data_root = REPO_ROOT / manifest["data_root"]
    video_subdir = Path(manifest["video_subdir"])
    output_size = tuple(manifest["output_size"])
    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(manifest["jpeg_quality"])]

    # Group frames by source video so that every video is opened only once.
    frames_by_video = defaultdict(list)
    for frame in manifest["frames"]:
        frames_by_video[(frame["take_name"], frame["camera"])].append(frame)

    written, skipped, missing = 0, 0, []
    for (take_name, camera), frames in tqdm(sorted(frames_by_video.items()), desc=manifest_path.stem):
        video_path = resolve_take_dir(egoexo_root, take_name) / video_subdir / camera
        if not video_path.is_file():
            missing.extend(f["output"] for f in frames)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            missing.extend(f["output"] for f in frames)
            continue

        for frame in sorted(frames, key=lambda f: f["frame_index"]):
            out_path = data_root / frame["output"]
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_index"])
            ok, image = cap.read()
            if not ok:
                missing.append(frame["output"])
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_path), image, jpg_params)
            written += 1
        cap.release()

    return written, skipped, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--egoexo_root", type=Path, required=True,
                        help="Directory containing the Ego-Exo4D take folders (e.g. <download_dir>/takes).")
    parser.add_argument("--split", choices=["all", "benchmark", "train"], default="all",
                        help="Which frames to extract: benchmark (evaluation), train (GRPO training) or all.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite frames that already exist.")
    args = parser.parse_args()

    splits = ["benchmark", "train"] if args.split == "all" else [args.split]
    all_missing = []
    for split in splits:
        written, skipped, missing = extract_split(MANIFESTS[split], args.egoexo_root, args.overwrite)
        print(f"[{split}] written: {written}, already present: {skipped}, failed: {len(missing)}")
        all_missing.extend(missing)

    if all_missing:
        print("\nThe following frames could not be extracted (missing takes/videos?):")
        for path in all_missing[:20]:
            print(f"  {path}")
        if len(all_missing) > 20:
            print(f"  ... and {len(all_missing) - 20} more")
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
