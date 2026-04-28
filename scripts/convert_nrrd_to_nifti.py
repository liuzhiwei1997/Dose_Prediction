#!/usr/bin/env python3
"""Convert a NRRD/SEG.NRRD image to NIfTI (.nii.gz)."""

from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def convert_nrrd_to_nifti(input_path: Path, output_path: Path) -> None:
    """Read a NRRD file and write it as NIfTI while preserving image metadata."""
    image = sitk.ReadImage(str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a NRRD/SEG.NRRD image to .nii.gz."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .nrrd or .seg.nrrd file path.",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output .nii.gz path. Defaults to input file stem + .nii.gz.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path: Path
    if args.output is not None:
        output_path = args.output
    else:
        stem = input_path.name
        for suffix in (".seg.nrrd", ".nrrd"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        output_path = input_path.with_name(f"{stem}.nii.gz")

    convert_nrrd_to_nifti(input_path, output_path)
    print(f"Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
