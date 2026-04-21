from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DosePrediction.Train.train_light_pyfer import Pyfer, TestOpenKBPDataModule
from DosePrediction.utils.runtime import get_lightning_accelerator


def build_model_config(best: dict[str, Any]) -> dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and export test metrics to Excel.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .ckpt file, or a directory containing last.ckpt",
    )
    parser.add_argument(
        "--best-json",
        default=None,
        help="Optional dvh_tuning_results.json to rebuild config for checkpoint loading",
    )
    parser.add_argument(
        "--output-xlsx",
        default="runs/DosePrediction/final_from_best_dvh_stage4/test_metrics_summary.xlsx",
        help="Output Excel path",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show test progress/logs in terminal")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (REPO_ROOT / path).resolve()
    return repo_candidate


def resolve_checkpoint(path_text: str) -> Path:
    raw = resolve_path(path_text)
    if raw.is_file():
        return raw
    if raw.is_dir():
        last_ckpt = raw / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt
    raise FileNotFoundError(f"Checkpoint not found: {raw}")


def load_best_json(best_json_text: str | None) -> dict[str, Any] | None:
    if not best_json_text:
        return None

    best_json = resolve_path(best_json_text)
    if not best_json.exists():
        raise FileNotFoundError(f"best-json not found: {best_json}")

    payload = json.loads(best_json.read_text(encoding="utf-8"))
    best = payload.get("best")
    if best is None:
        raise ValueError(f"Missing 'best' key in: {best_json}")
    return best


def resolve_model(checkpoint_path: Path, best: dict[str, Any] | None) -> Pyfer:
    if best is not None:
        model_cfg = build_model_config(best)
        try:
            return Pyfer.load_from_checkpoint(str(checkpoint_path), config_param=model_cfg, freeze=False)
        except Exception as exc:
            print(f"[WARN] Loading with --best-json config failed, retrying default loader: {exc}")

    return Pyfer.load_from_checkpoint(str(checkpoint_path))


def summarize_structures(per_patient: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_patient.columns if c != "patient"]
    numeric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(per_patient[c])]
    if not numeric_cols:
        return pd.DataFrame(columns=["metric", "mean", "std", "min", "max"])

    rows = []
    for col in numeric_cols:
        vals = per_patient[col].dropna()
        if len(vals) == 0:
            continue
        rows.append(
            {
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    best_payload = load_best_json(args.best_json)

    output_path = resolve_path(args.output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = resolve_model(checkpoint_path, best_payload)
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
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
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
                "best_json_used": "yes" if best_payload is not None else "no",
            }
        ]
    )

    structure_summary = summarize_structures(per_patient)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        per_patient.to_excel(writer, sheet_name="per_patient", index=False)
        structure_summary.to_excel(writer, sheet_name="metric_stats", index=False)

    print(f"Excel generated: {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
