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
from pytorch_lightning.callbacks import ModelCheckpoint

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
            metric_name = None
            for metric in metric_diffs.keys():
                if key.endswith(f"_{metric}"):
                    metric_name = metric
                    break
            if metric_name is None:
                continue
            gt_key = "gt_" + key[3:]
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


def progress_file(output_root: str) -> Path:
    return Path(output_root) / "dvh_tuning_progress.json"


def summary_file(output_root: str) -> Path:
    return Path(output_root) / "dvh_tuning_results.json"


def load_progress(output_root: str) -> Dict[str, Dict[str, float]]:
    path = progress_file(output_root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("all_results", [])
    return {str(r["trial_id"]): r for r in results if "trial_id" in r}


def save_progress(output_root: str, results: Dict[str, Dict[str, float]]) -> None:
    ordered = [results[k] for k in sorted(results.keys(), key=lambda x: int(x))]
    if not ordered:
        return
    ranked = sorted(ordered, key=lambda x: x["objective"])
    payload = {
        "best": ranked[0],
        "all_results": ranked,
        "completed_trials": len(ordered),
        "best_trial_id": ranked[0].get("trial_id"),
    }
    progress_path = progress_file(output_root)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_file(output_root).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prune_non_best_trial_dirs(output_root: str, best_trial_id: int | str) -> None:
    root = Path(output_root)
    keep_dir = f"trial_{int(best_trial_id):03d}"
    for trial_dir in root.glob("trial_*"):
        if trial_dir.name != keep_dir and trial_dir.is_dir():
            shutil.rmtree(trial_dir, ignore_errors=True)


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
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(trial_dir),
        save_last=True,
        every_n_epochs=max(1, args.checkpoint_every_n_epochs),
        save_top_k=args.checkpoint_top_k,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=max(1, args.check_val_every_n_epoch),
        logger=build_logger(run_name=f"dvh_tune_trial_{trial_id}"),
        default_root_dir=str(trial_dir),
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )
    last_ckpt = trial_dir / "last.ckpt"
    resume_ckpt = str(last_ckpt) if (args.resume_trials and last_ckpt.exists()) else None
    if resume_ckpt:
        print(f"[INFO] Trial {trial_id}: resuming from {resume_ckpt}")
    trainer.fit(model, datamodule=train_val_data, ckpt_path=resume_ckpt)
    trainer.test(model, datamodule=test_data)

    dose_score = float(np.mean(model.list_dose_metric))
    dvh_score = float(np.mean(model.list_DVH_dif))
    per_metric = aggregate_metric_mae(model.dict_DVH_dif)

    objective = dvh_score + args.dose_weight * dose_score
    result = {
        "trial_id": trial_id,
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
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=None,
        help="Validation/test cadence during fit. Defaults to max-epochs (validate once at the end).",
    )
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=10,
        help="How often to save epoch checkpoints inside each trial directory.",
    )
    parser.add_argument(
        "--checkpoint-top-k",
        type=int,
        default=1,
        help="How many periodic checkpoints to keep per trial (-1 keeps all and uses more disk).",
    )
    parser.add_argument("--dose-weight", type=float, default=0.5)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit number of grid trials (e.g., 20).",
    )
    parser.add_argument(
        "--resume-trials",
        action="store_true",
        default=True,
        help="Resume interrupted trial from <trial_dir>/last.ckpt when available.",
    )
    parser.add_argument(
        "--no-resume-trials",
        dest="resume_trials",
        action="store_false",
    )
    parser.add_argument("--cleanup-trials", action="store_true")
    parser.add_argument(
        "--keep-only-best-trial",
        action="store_true",
        help="Delete non-best trial_* checkpoint folders to save disk during/after tuning.",
    )
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
    if args.check_val_every_n_epoch is None:
        args.check_val_every_n_epoch = max(1, args.max_epochs)

    os.makedirs(args.output_root, exist_ok=True)
    existing = load_progress(args.output_root)
    if existing:
        print(f"[INFO] Found {len(existing)} completed trial(s), will skip them.")
    results: Dict[str, Dict[str, float]] = dict(existing)
    if args.keep_only_best_trial and results:
        current_best = min(results.values(), key=lambda x: x["objective"])
        prune_non_best_trial_dirs(args.output_root, current_best["trial_id"])

    grid = list(get_trial_grid(args))
    if args.max_trials is not None:
        if args.max_trials <= 0:
            raise ValueError("--max-trials must be > 0")
        grid = grid[:args.max_trials]

    for idx, trial_cfg in enumerate(grid, start=1):
        if str(idx) in results:
            print(f"[INFO] Skip completed Trial {idx}")
            continue
        print(f"\n=== Trial {idx}: {trial_cfg} ===")
        result = run_trial(idx, trial_cfg, args)
        results[str(idx)] = result
        save_progress(args.output_root, results)
        if args.keep_only_best_trial:
            current_best = min(results.values(), key=lambda x: x["objective"])
            prune_non_best_trial_dirs(args.output_root, current_best["trial_id"])
        print(
            f"objective={result['objective']:.4f}, dose={result['dose_score']:.4f}, "
            f"dvh={result['dvh_score']:.4f}, D0.1cc={result['mae_D_0.1_cc']:.4f}, D99={result['mae_D99']:.4f}"
        )

    if not results:
        raise RuntimeError("No trials were executed. Check your grid/max-trials settings.")

    ordered_results = [results[k] for k in sorted(results.keys(), key=lambda x: int(x))]
    ranked_results = sorted(ordered_results, key=lambda x: x["objective"])
    best = ranked_results[0]
    summary_path = summary_file(args.output_root)
    summary_path.write_text(json.dumps({"best": best, "all_results": ranked_results}, indent=2), encoding="utf-8")
    if args.keep_only_best_trial:
        prune_non_best_trial_dirs(args.output_root, best["trial_id"])

    print("\nBest config:")
    print(json.dumps(best, indent=2))
    print(f"\nSaved detailed results to: {summary_path}")


if __name__ == "__main__":
    main()
