"""Run dense optical flow on a pair of sonar frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from debris_flow.optical_flow import FarnebackConfig, run_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame1", type=Path, required=True, help="First frame path.")
    parser.add_argument("--frame2", type=Path, required=True, help="Second frame path.")
    parser.add_argument("--output", type=Path, default=Path("runs/pair_demo"))
    parser.add_argument("--winsize", type=int, default=15)
    parser.add_argument("--threshold-std-factor", type=float, default=0.5)
    parser.add_argument("--roi-top", type=float, default=0.0)
    parser.add_argument("--roi-bottom", type=float, default=0.0)
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument("--direction-correction", action="store_true")
    parser.add_argument("--region-expansion", action="store_true")
    parser.add_argument("--annotate-boxes", action="store_true")
    parser.add_argument("--box-min-area", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FarnebackConfig(
        winsize=args.winsize,
        threshold_std_factor=args.threshold_std_factor,
        enhance_brightness=not args.no_enhance,
        direction_correction=args.direction_correction,
        region_expansion=args.region_expansion,
    )

    result = run_pair(
        frame1_path=args.frame1,
        frame2_path=args.frame2,
        output_dir=args.output,
        config=config,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
        annotate_boxes=args.annotate_boxes,
        box_min_area=args.box_min_area,
    )

    print(f"Optical flow completed in {result['elapsed_ms']:.2f} ms")
    print(f"Dense flow image: {result['flow_image']}")
    print(f"Motion-vector image: {result['arrow_image']}")
    if args.annotate_boxes:
        print(f"Detected boxes: {len(result['boxes'])}")


if __name__ == "__main__":
    main()
