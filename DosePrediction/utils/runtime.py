from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

import importlib
import warnings

import torch


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = get_repo_root()
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("DOSE_PREDICTION_OUTPUT_ROOT", REPO_ROOT / "runs"))
DEFAULT_DATA_ROOT = Path(os.environ.get("DOSE_PREDICTION_DATA_ROOT", REPO_ROOT))


def ensure_dir(path: os.PathLike | str) -> str:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return str(path_obj)


def resolve_repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def resolve_data_root() -> str:
    return str(DEFAULT_DATA_ROOT)


def resolve_output_dir(*parts: str) -> str:
    return ensure_dir(DEFAULT_OUTPUT_ROOT.joinpath(*parts))


def resolve_optional_checkpoint(env_var: str, *repo_candidates: str) -> Optional[str]:
    env_value = os.environ.get(env_var)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return str(candidate)

    for candidate in repo_candidates:
        path = REPO_ROOT.joinpath(candidate)
        if path.exists():
            return str(path)
    return None


def get_lightning_accelerator() -> Tuple[str, int]:
    if torch.cuda.is_available() and os.environ.get("DOSE_PREDICTION_FORCE_CPU", "0") != "1":
        return "gpu", 1
    return "cpu", 1


def use_pin_memory() -> bool:
    return torch.cuda.is_available() and os.environ.get("DOSE_PREDICTION_FORCE_CPU", "0") != "1"


def use_bitsandbytes() -> bool:
    return os.environ.get("DOSE_PREDICTION_USE_BNB", "0") == "1"


@lru_cache(maxsize=1)
def get_bitsandbytes_module():
    if not use_bitsandbytes():
        return None

    try:
        return importlib.import_module("bitsandbytes")
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        warnings.warn(
            f"bitsandbytes is unavailable and will be disabled: {exc}",
            RuntimeWarning,
        )
        return None
