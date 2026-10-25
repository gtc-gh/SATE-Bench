"""
Collects every results/<setting>/**/metrics.json into one table in the layout of Table 4.

    python evaluation/summarize.py                 # prints a markdown table
    python evaluation/summarize.py --csv summary.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from common import REPO_ROOT  # noqa: F401  (adds the repository root to sys.path)
from sate_bench.config import RESULTS_ROOT, TASKS

ABBREV = ["HC", "OC", "FAP", "HOI", "HE", "OE", "RD-H", "RD-O", "VDO", "VSD"]
TASK_ABBREV = dict(zip(TASKS, ABBREV))


def collect(results_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_root.rglob("metrics.json")):
        m = json.loads(path.read_text())
        if "setting" not in m or not m.get("total_samples"):
            continue
        # Runs written with a custom --results_dir may not live under results/<setting>/.
        setting_root = results_root / m["setting"]
        run_dir = path.parent
        run = run_dir.relative_to(setting_root if run_dir.is_relative_to(setting_root) else results_root)
        row = {"setting": m["setting"], "run": str(run), "N": m["total_samples"]}
        for task, abbr in TASK_ABBREV.items():
            row[abbr] = m.get("per_task", {}).get(task, {}).get("accuracy", float("nan")) * 100
        row["Avg. Acc"] = m["accuracy"] * 100
        if m["setting"] == "active":
            row["Avg. Steps"] = m["avg_steps"]
            row["Avg. Reward"] = m["avg_reward"] * 100
            row["Active Rate"] = m["active_rate"] * 100
        row["Time (h)"] = m.get("inference_time_hours", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    df = collect(args.results_root)
    if df.empty:
        print(f"No metrics.json found under {args.results_root}")
        return
    for setting in ("active", "passive"):
        sub = df[df["setting"] == setting].drop(columns="setting").dropna(axis=1, how="all")
        if not sub.empty:
            print(f"\n### {setting.capitalize()} setting\n")
            print(sub.to_markdown(index=False, floatfmt=".1f"))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
