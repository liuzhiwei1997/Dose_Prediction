from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


def to_float(value: str) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def load_metrics(metrics_path: Path) -> List[Dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def find_best_row(rows: List[Dict[str, str]], metric_col: str, mode: str) -> Dict[str, str]:
    valid_rows: List[Dict[str, str]] = []
    for row in rows:
        val = to_float(row.get(metric_col, ""))
        if val is not None:
            valid_rows.append(row)
    if not valid_rows:
        raise RuntimeError(f"No valid '{metric_col}' values in metrics file.")

    reverse = mode == "max"
    return sorted(valid_rows, key=lambda r: to_float(r.get(metric_col, "")), reverse=reverse)[0]


def infer_epoch_from_step(rows: List[Dict[str, str]], step_value: str) -> str:
    if step_value in (None, ""):
        return "N/A"
    for row in rows:
        if row.get("step", "") == step_value and row.get("epoch", "") not in ("", None):
            return row.get("epoch", "N/A")
    return "N/A"


def parse_ckpt_epoch_step(ckpt_name: str) -> Optional[tuple]:
    match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", ckpt_name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find best epoch from Lightning CSV metrics.")
    parser.add_argument("--logs-root", default="runs/logs/dose_prediction", help="Folder containing metrics.csv")
    parser.add_argument("--metric", default="mean_dose_score", help="Metric column to optimize")
    parser.add_argument("--mode", choices=["max", "min"], default="max", help="Optimization direction")
    parser.add_argument("--checkpoint-dir", default="runs/DosePrediction/final", help="Directory with checkpoints")
    args = parser.parse_args()

    metrics_files = sorted(Path(args.logs_root).glob("**/metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.csv found under: {args.logs_root}")

    metrics_path = metrics_files[-1]
    rows = load_metrics(metrics_path)
    best = find_best_row(rows, args.metric, args.mode)
    best_epoch = best.get("epoch", "")
    if best_epoch in ("", None):
        best_epoch = infer_epoch_from_step(rows, best.get("step", ""))

    print(f"metrics_file: {metrics_path}")
    print(f"metric: {args.metric} ({args.mode})")
    print(f"best_epoch: {best_epoch if best_epoch not in ('', None) else 'N/A'}")
    print(f"best_step: {best.get('step', 'N/A')}")
    print(f"best_metric: {best.get(args.metric, 'N/A')}")
    if args.metric == "mean_dose_score":
        metric_val = to_float(best.get(args.metric, ""))
        if metric_val is not None:
            print(f"estimated_dose_score: {-metric_val:.6f}")
    if "val_loss" in best and best["val_loss"] != "":
        print(f"val_loss: {best['val_loss']}")

    ckpt_dir = Path(args.checkpoint_dir)
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        print(f"checkpoint_count: {len(ckpts)}")
        if (ckpt_dir / "last.ckpt").exists():
            print(f"last_ckpt: {ckpt_dir / 'last.ckpt'}")
        if ckpts:
            print("recent_ckpts:")
            for ck in ckpts[-5:]:
                print(f"  - {ck}")

        best_step = to_float(best.get("step", ""))
        named_ckpts = [parse_ckpt_epoch_step(ck.name) for ck in ckpts]
        named_ckpts = [c for c in named_ckpts if c is not None]
        if best_step is not None and named_ckpts:
            step_values = [s for _, s in named_ckpts]
            if best_step < min(step_values) or best_step > max(step_values):
                print("[WARN] metrics step range and checkpoint step range do not overlap.")
                print("[WARN] This usually means logs-root and checkpoint-dir are from different runs.")


if __name__ == "__main__":
    main()
