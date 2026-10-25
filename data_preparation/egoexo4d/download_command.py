"""
Prints the `egoexo` command that downloads only the Ego-Exo4D videos used by SATE-Bench.

The Ego-Exo4D downloader selects takes by `take_uid`, while SATE-Bench lists take names, so this
script looks the names up in Ego-Exo4D's `takes.json` (downloaded with `--parts metadata`).

    egoexo -o <EGOEXO_DIR> --parts metadata
    python data_preparation/egoexo4d/download_command.py --takes_json <EGOEXO_DIR>/takes.json
"""

import argparse
import json
from pathlib import Path

from extract_frames import MANIFESTS


def required_takes(split: str) -> list:
    splits = ["benchmark", "train"] if split == "all" else [split]
    names = set()
    for s in splits:
        names |= {f["take_name"] for f in json.loads(MANIFESTS[s].read_text())["frames"]}
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--takes_json", type=Path, required=True, help="Ego-Exo4D metadata file takes.json.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Download directory for egoexo (default: the folder containing takes.json).")
    parser.add_argument("--split", choices=["all", "benchmark", "train"], default="all",
                        help="Takes needed for the benchmark, the GRPO training data, or both.")
    args = parser.parse_args()

    takes = json.loads(args.takes_json.read_text())
    uid_of = {t["take_name"]: t["take_uid"] for t in takes}
    names = required_takes(args.split)
    missing = [n for n in names if n not in uid_of]
    if missing:
        raise SystemExit(f"{len(missing)} takes were not found in {args.takes_json}: {', '.join(missing[:5])} ...")

    out_dir = args.out_dir or args.takes_json.parent
    uids = " ".join(uid_of[n] for n in names)
    print(f"# {len(names)} Ego-Exo4D takes needed for split '{args.split}'. Run:\n")
    print(f"egoexo -o {out_dir} --parts downscaled_takes/448 --uids {uids}")


if __name__ == "__main__":
    main()
