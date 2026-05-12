"""Compute a frequency-spectrum visualization for a sonar frame."""

from __future__ import annotations

import argparse
from pathlib import Path

from debris_flow.spectral_analysis import save_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input sonar frame.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/spectrum/magnitude_spectrum.png"),
        help="Output spectrum image.",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("runs/spectrum/comparison.png"),
        help="Optional side-by-side original/spectrum image. Use 'none' to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_path = None if str(args.comparison).lower() == "none" else args.comparison
    spectrum_path, comparison = save_spectrum(args.input, args.output, comparison_path)

    print(f"Spectrum image: {spectrum_path}")
    if comparison is not None:
        print(f"Comparison image: {comparison}")


if __name__ == "__main__":
    main()
