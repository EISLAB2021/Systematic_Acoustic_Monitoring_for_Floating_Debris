"""Project a rectangular sonar frame into a fan-shaped image."""

from __future__ import annotations

import argparse
from pathlib import Path

from debris_flow.sonar_projection import ProjectionConfig, save_fan_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/sonar_fan_projection.png"))
    parser.add_argument("--start-row", type=int, default=400)
    parser.add_argument("--end-row", type=int, default=1100)
    parser.add_argument("--range-min", type=float, default=0.971)
    parser.add_argument("--range-resolution", type=float, default=0.0029)
    parser.add_argument("--beam-width-deg", type=float, default=30.0)
    parser.add_argument("--scaled-width", type=int, default=128)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--sector-edge-deg", type=float, default=75.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectionConfig(
        start_row=args.start_row,
        end_row=args.end_row,
        range_min=args.range_min,
        range_resolution=args.range_resolution,
        beam_width_deg=args.beam_width_deg,
        scaled_width=args.scaled_width,
        rotation_deg=args.rotation_deg,
        sector_edge_deg=args.sector_edge_deg,
    )
    output_path = save_fan_projection(args.input, args.output, config)
    print(f"Saved fan projection: {output_path}")


if __name__ == "__main__":
    main()
