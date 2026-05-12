"""Generate YOLO pseudo-labels from flow motion images."""

from __future__ import annotations

import argparse
from pathlib import Path

from debris_flow.pseudo_label import LabelConfig, generate_pseudo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-images", type=Path, required=True)
    parser.add_argument("--sonar-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sonar_pseudo_labels"))
    parser.add_argument("--csv-path", type=Path, default=Path("runs/sonar_pseudo_labels.csv"))
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--debris-threshold", type=int, default=10)
    parser.add_argument("--min-width", type=int, default=50)
    parser.add_argument("--min-height", type=int, default=50)
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--min-area", type=int, default=80 * 80)
    parser.add_argument("--class-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabelConfig(
        debris_threshold=args.debris_threshold,
        min_width=args.min_width,
        min_height=args.min_height,
        margin=args.margin,
        min_area=args.min_area,
        class_id=args.class_id,
    )
    rows = generate_pseudo_labels(
        flow_images_dir=args.flow_images,
        sonar_images_dir=args.sonar_images,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        pattern=args.pattern,
        config=config,
    )
    print(f"Saved {len(rows)} labels to {args.output_dir}")
    print(f"CSV summary: {args.csv_path}")


if __name__ == "__main__":
    main()
