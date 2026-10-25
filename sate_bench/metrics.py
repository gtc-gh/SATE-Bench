"""Metrics and resumable result recording.

Active setting (Sec. 3.4 of the paper), for each question i answered after s_i steps:
    accuracy     = mean_i I(c_i)
    avg_steps    = mean_i s_i
    reward       = I(c_i) * exp(-lambda * s_i)          (lambda = 0.1)
    active       = 1 if the model requested at least one new view, else 0

The paper reports Avg. Acc and Avg. Steps averaged over all questions, and Avg. Reward and
Active Rate averaged over the ten task categories. Both variants are written to metrics.json.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import LAMBDA_VAL

ACTIVE_COLUMNS = ["index", "question_type", "question_name", "turn", "view", "raw_output", "thought_process",
                  "action", "pred_answer", "gt_answer", "correct"]
PASSIVE_COLUMNS = ["index", "question_type", "question_name", "raw_output", "thought_process",
                   "pred_answer", "gt_answer", "correct"]


def step_reward(correct: int, steps: int, lam: float = LAMBDA_VAL) -> float:
    return math.exp(-lam * steps) if correct else 0.0


def summarize_active(records: List[dict]) -> dict:
    df = pd.DataFrame(records)
    if df.empty:
        return {"total_samples": 0}
    per_task = df.groupby("question_name").agg(
        num_samples=("correct", "size"), accuracy=("correct", "mean"), avg_steps=("turns", "mean"),
        reward=("reward", "mean"), active_rate=("explored", "mean"))
    return {
        "total_samples": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "avg_steps": float(df["turns"].mean()),
        "avg_reward": float(per_task["reward"].mean()),             # mean over tasks (as in the paper)
        "active_rate": float(per_task["active_rate"].mean()),       # mean over tasks (as in the paper)
        "avg_reward_sample_mean": float(df["reward"].mean()),
        "active_rate_sample_mean": float(df["explored"].mean()),
        "per_task": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_task.to_dict("index").items()},
    }


def summarize_passive(records: List[dict]) -> dict:
    df = pd.DataFrame(records)
    if df.empty:
        return {"total_samples": 0}
    per_task = df.groupby("question_name").agg(num_samples=("correct", "size"), accuracy=("correct", "mean"))
    return {
        "total_samples": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "per_task": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_task.to_dict("index").items()},
    }


class ResultRecorder:
    """Appends per-sample rows to predictions.csv and keeps metrics.json up to date.

    Re-running the same command resumes from the samples already stored in predictions.csv.
    """

    def __init__(self, results_dir, setting: str, run_info: Dict):
        self.setting = setting
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.results_dir / "predictions.csv"
        self.json_path = self.results_dir / "metrics.json"
        self.columns = ACTIVE_COLUMNS if setting == "active" else PASSIVE_COLUMNS
        self.run_info = run_info
        self.records: List[dict] = []
        self.done = set()
        self.prev_time = 0.0
        self.start_time = time.time()
        self._load_existing()

    def _load_existing(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            pd.DataFrame(columns=self.columns).to_csv(self.csv_path, index=False)
            return
        df = pd.read_csv(self.csv_path)
        for idx, group in df.groupby("index", sort=False):
            self.done.add(int(idx))
            self.records.append(self._record_from_rows(group.to_dict("records")))
        if self.json_path.exists():
            self.prev_time = json.loads(self.json_path.read_text()).get("inference_time_hours", 0.0) * 3600
        print(f"Resuming: {len(self.done)} samples already evaluated in {self.csv_path}")

    def _record_from_rows(self, rows: List[dict]) -> dict:
        last = rows[-1]
        correct = int(last["correct"] == 1)
        record = {"index": int(last["index"]), "question_name": last["question_name"], "correct": correct}
        if self.setting == "active":
            turns = int(max(r["turn"] for r in rows))
            record.update(turns=turns, reward=step_reward(correct, turns),
                          explored=int(any(r["action"] in ("X", "Y") for r in rows)))
        return record

    def add(self, rows: List[dict]):
        """Stores all rows (turns) of one finished sample."""
        df = pd.DataFrame(rows, columns=self.columns)
        df["correct"] = df["correct"].astype("Int64")
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
        self.done.add(int(rows[0]["index"]))
        self.records.append(self._record_from_rows(rows))
        self.write_metrics()

    def write_metrics(self):
        summary = summarize_active(self.records) if self.setting == "active" else summarize_passive(self.records)
        elapsed = self.prev_time + time.time() - self.start_time
        out = {"setting": self.setting, **self.run_info, **summary, "inference_time_hours": elapsed / 3600.0}
        self.json_path.write_text(json.dumps(out, indent=2))
        return out
