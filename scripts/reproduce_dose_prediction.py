from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path



def write_nifti(array, output_path: Path) -> None:
    import numpy as np
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(output_path))


def create_mock_patient(patient_dir: Path, image_size: int) -> None:
    import numpy as np

    grid = np.indices((image_size, image_size, image_size), dtype=np.float32)
    zz, yy, xx = grid

    center = (image_size - 1) / 2.0
    radius = max(image_size / 6.0, 2.0)

    ct = (xx + yy + zz) / (3 * image_size)
    dose = np.clip(1.2 - (((xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2) ** 0.5) / image_size, 0, 1)
    dose *= 70.0
    dose_mask = np.ones_like(dose, dtype=np.float32)

    structure_offsets = {
        "Brainstem": (-3, 0, 0),
        "SpinalCord": (3, 0, 0),
        "RightParotid": (0, -4, 0),
        "LeftParotid": (0, 4, 0),
        "Esophagus": (0, 0, -4),
        "Larynx": (0, 0, 4),
        "Mandible": (0, 0, 0),
        "PTV70": (0, 0, 0),
        "PTV63": (2, 2, 2),
        "PTV56": (-2, -2, -2),
    }

    write_nifti(ct, patient_dir / "CT.nii.gz")
    write_nifti(dose, patient_dir / "dose.nii.gz")
    write_nifti(dose_mask, patient_dir / "possible_dose_mask.nii.gz")

    for name, (dx, dy, dz) in structure_offsets.items():
        shifted = ((xx - (center + dx)) ** 2 + (yy - (center + dy)) ** 2 + (zz - (center + dz)) ** 2) <= radius ** 2
        write_nifti(shifted.astype(np.float32), patient_dir / f"{name}.nii.gz")


def build_mock_dataset(root: Path, image_size: int) -> None:
    if root.exists():
        shutil.rmtree(root)

    train_dir = root / "provided-data" / "nifti-train-pats" / "pt_001"
    val_dir = root / "provided-data" / "nifti-test-pats" / "pt_101"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    create_mock_patient(train_dir, image_size=image_size)
    create_mock_patient(val_dir, image_size=image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny mock OpenKBP dataset and run a smoke test.")
    parser.add_argument("--workspace", default=".cache/mock_repro", help="Temporary workspace for generated data.")
    parser.add_argument("--image-size", type=int, default=32, help="Synthetic 3D volume edge length.")
    parser.add_argument("--max-epochs", type=int, default=1, help="Trainer max epochs for the smoke test.")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    build_mock_dataset(workspace, image_size=args.image_size)

    os.environ.setdefault("DOSE_PREDICTION_DATA_ROOT", str(workspace))
    os.environ.setdefault("DOSE_PREDICTION_OUTPUT_ROOT", str(workspace / "runs"))
    os.environ.setdefault("DOSE_PREDICTION_IMAGE_SIZE", str(args.image_size))
    os.environ.setdefault("DOSE_PREDICTION_BATCH_SIZE", "1")
    os.environ.setdefault("DOSE_PREDICTION_SW_BATCH_SIZE", "1")
    os.environ.setdefault("DOSE_PREDICTION_NUM_WORKERS", "0")
    os.environ.setdefault("DOSE_PREDICTION_LOGGER", "csv")
    os.environ.setdefault("DOSE_PREDICTION_FORCE_CPU", "1")
    os.environ.setdefault("DOSE_PREDICTION_USE_BNB", "0")

    from DosePrediction.Train.train_light_pyfer import main as run_training

    run_training(
        freeze=False,
        max_epochs=args.max_epochs,
        fast_dev_run=True,
    )

    print(f"Smoke test finished successfully. Mock data root: {workspace}")


if __name__ == "__main__":
    main()
