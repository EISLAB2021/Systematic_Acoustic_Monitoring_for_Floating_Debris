"""Build optical-flow visualizations for a sorted sonar image sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

from debris_flow.optical_flow import FarnebackConfig, build_flow_image_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sonar_flow_images"))
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--winsize", type=int, default=12)
    parser.add_argument("--down-threshold", type=float, default=1.0)
    parser.add_argument("--dbscan-eps", type=float, default=25.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=2)
    parser.add_argument("--no-enhance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FarnebackConfig(
        winsize=args.winsize,
        enhance_brightness=not args.no_enhance,
    )
    outputs = build_flow_image_sequence(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        config=config,
        down_threshold=args.down_threshold,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )
    print(f"Saved {len(outputs)} flow images to {args.output_dir}")


if __name__ == "__main__":
    main()
