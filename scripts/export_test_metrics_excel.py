from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from DosePrediction.Train.train_light_pyfer import Pyfer, TestOpenKBPDataModule
from DosePrediction.utils.runtime import get_lightning_accelerator


def build_model_config(best: dict) -> dict:
    return {
        "act": best.get("act", "mish"),
        "multiS_conv": bool(best.get("multiS_conv", True)),
        "lr": float(best.get("lr", 0.0006130697604327541)),
        "weight_decay": float(best.get("weight_decay", 0.00016303111017674179)),
        "delta1": float(best.get("delta1", 10.0)),
        "delta2": float(best.get("delta2", 8.0)),
        "hotspot_weight": float(best.get("hotspot_weight", 0.75)),
        "hotspot_quantile": float(best.get("hotspot_quantile", 0.98)),
        "coldspot_weight": float(best.get("coldspot_weight", 0.35)),
        "coldspot_quantile": float(best.get("coldspot_quantile", 0.10)),
    }


def resolve_model(checkpoint_path: Path, best_json: Path | None) -> Pyfer:
    if best_json is None:
        return Pyfer.load_from_checkpoint(str(checkpoint_path))

    payload = json.loads(best_json.read_text(encoding="utf-8"))
    best = payload.get("best")
    if best is None:
        raise ValueError(f"Missing 'best' key in: {best_json}")

    model_cfg = build_model_config(best)
    return Pyfer.load_from_checkpoint(str(checkpoint_path), config_param=model_cfg, freeze=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and export test metrics to Excel.")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--best-json",
        default=None,
        help="Optional DVH tuning summary json (dvh_tuning_results.json) to rebuild config for checkpoint loading",
    )
    parser.add_argument(
        "--output-xlsx",
        default="runs/DosePrediction/final_from_best_dvh_stage4/test_metrics_summary.xlsx",
        help="Output Excel path",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show test progress/logs in terminal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    best_json = Path(args.best_json) if args.best_json else None
    if best_json is not None and not best_json.exists():
        raise FileNotFoundError(f"best-json not found: {best_json}")

    output_path = Path(args.output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = resolve_model(checkpoint_path, best_json)
    data = TestOpenKBPDataModule()

    accelerator, devices = get_lightning_accelerator()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=args.show_progress,
    )

    if args.show_progress:
        trainer.test(model, datamodule=data)
    else:
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
            trainer.test(model, datamodule=data)

    per_patient = pd.DataFrame.from_dict(model.dict_DVH_dif, orient="index")
    per_patient.index.name = "patient"
    per_patient = per_patient.reset_index()

    mean_dose = float(np.mean(model.list_dose_metric)) if model.list_dose_metric else float("nan")
    mean_dvh = float(np.mean(model.list_DVH_dif)) if model.list_DVH_dif else float("nan")

    summary = pd.DataFrame(
        [
            {
                "num_patients": int(len(per_patient)),
                "mean_dose_score": mean_dose,
                "mean_dvh_score": mean_dvh,
                "checkpoint": str(checkpoint_path),
                "best_json": str(best_json) if best_json else "",
            }
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        per_patient.to_excel(writer, sheet_name="per_patient", index=False)

    print(f"Excel generated: {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
