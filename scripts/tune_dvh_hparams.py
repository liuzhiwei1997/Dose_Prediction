from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pytorch_lightning as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import DosePrediction.Train.config as config
from DosePrediction.Train.train_light_pyfer import OpenKBPDataModule, TestOpenKBPDataModule, Pyfer, build_logger
from DosePrediction.utils.runtime import get_lightning_accelerator


def parse_csv_float(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def aggregate_metric_mae(dict_dvh: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    metric_diffs: Dict[str, List[float]] = {
        "D1": [],
        "D95": [],
        "D99": [],
        "D_0.1_cc": [],
        "mean": [],
    }
    for patient_values in dict_dvh.values():
        for key, pred_value in patient_values.items():
            if not key.startswith("pre"):
                continue
            metric_name = key.rsplit("_", 1)[-1]
            if metric_name not in metric_diffs:
                continue
            gt_key = "gt" + key[3:]
            if gt_key not in patient_values:
                continue
            metric_diffs[metric_name].append(abs(pred_value - patient_values[gt_key]))

    return {
        metric_name: (float(np.mean(values)) if values else float("nan"))
        for metric_name, values in metric_diffs.items()
    }


def get_trial_grid(args: argparse.Namespace) -> Iterable[Dict[str, float]]:
    for hotspot_weight, coldspot_weight, hotspot_q, coldspot_q in itertools.product(
            parse_csv_float(args.hotspot_weights),
            parse_csv_float(args.coldspot_weights),
            parse_csv_float(args.hotspot_quantiles),
            parse_csv_float(args.coldspot_quantiles),
    ):
        yield {
            "hotspot_weight": hotspot_weight,
            "coldspot_weight": coldspot_weight,
            "hotspot_quantile": hotspot_q,
            "coldspot_quantile": coldspot_q,
        }


def run_trial(
        trial_id: int,
        trial_cfg: Dict[str, float],
        args: argparse.Namespace,
) -> Dict[str, float]:
    trial_dir = Path(args.output_root) / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "act": args.act,
        "multiS_conv": args.multi_s_conv,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "delta1": args.delta1,
        "delta2": args.delta2,
        **trial_cfg,
    }
    model = Pyfer(cfg, freeze=args.freeze)
    train_val_data = OpenKBPDataModule()
    test_data = TestOpenKBPDataModule()

    accelerator, devices = get_lightning_accelerator()
    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=max(1, args.max_epochs),
        logger=build_logger(run_name=f"dvh_tune_trial_{trial_id}"),
        default_root_dir=str(trial_dir),
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=train_val_data)
    trainer.test(model, datamodule=test_data)

    dose_score = float(np.mean(model.list_dose_metric))
    dvh_score = float(np.mean(model.list_DVH_dif))
    per_metric = aggregate_metric_mae(model.dict_DVH_dif)

    objective = dvh_score + args.dose_weight * dose_score
    result = {
        **trial_cfg,
        "dose_score": dose_score,
        "dvh_score": dvh_score,
        "objective": objective,
        **{f"mae_{k}": v for k, v in per_metric.items()},
    }

    if args.cleanup_trials:
        shutil.rmtree(trial_dir, ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search DVH-oriented loss weights on validation set.")
    parser.add_argument("--output-root", default=str(Path(config.CHECKPOINT_RESULT_DIR) / "dvh_tuning"))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--dose-weight", type=float, default=0.5)
    parser.add_argument("--cleanup-trials", action="store_true")
    parser.add_argument("--freeze", action="store_true", default=True)
    parser.add_argument("--no-freeze", dest="freeze", action="store_false")

    parser.add_argument("--act", default="mish")
    parser.add_argument("--multi-s-conv", action="store_true", default=True)
    parser.add_argument("--no-multi-s-conv", dest="multi_s_conv", action="store_false")
    parser.add_argument("--lr", type=float, default=0.0006130697604327541)
    parser.add_argument("--weight-decay", type=float, default=0.00016303111017674179)
    parser.add_argument("--delta1", type=float, default=10.0)
    parser.add_argument("--delta2", type=float, default=8.0)

    parser.add_argument("--hotspot-weights", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--coldspot-weights", default="0.2,0.35,0.5,0.8")
    parser.add_argument("--hotspot-quantiles", default="0.975,0.98,0.99")
    parser.add_argument("--coldspot-quantiles", default="0.05,0.10,0.15")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    results: List[Dict[str, float]] = []

    for idx, trial_cfg in enumerate(get_trial_grid(args), start=1):
        print(f"\n=== Trial {idx}: {trial_cfg} ===")
        result = run_trial(idx, trial_cfg, args)
        results.append(result)
        print(
            f"objective={result['objective']:.4f}, dose={result['dose_score']:.4f}, "
            f"dvh={result['dvh_score']:.4f}, D0.1cc={result['mae_D_0.1_cc']:.4f}, D99={result['mae_D99']:.4f}"
        )

    results = sorted(results, key=lambda x: x["objective"])
    best = results[0]
    summary_path = Path(args.output_root) / "dvh_tuning_results.json"
    summary_path.write_text(json.dumps({"best": best, "all_results": results}, indent=2), encoding="utf-8")

    print("\nBest config:")
    print(json.dumps(best, indent=2))
    print(f"\nSaved detailed results to: {summary_path}")


if __name__ == "__main__":
    main()
