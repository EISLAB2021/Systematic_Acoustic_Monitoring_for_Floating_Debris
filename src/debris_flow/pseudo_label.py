"""Motion-based pseudo-label generation for underwater floating debris sonar frames."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .optical_flow import IMAGE_EXTENSIONS, list_images, read_image


@dataclass
class Box:
    """Pixel-space bounding box."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def clamp(self, width: int, height: int) -> "Box":
        return Box(
            xmin=max(0, min(self.xmin, width - 1)),
            ymin=max(0, min(self.ymin, height - 1)),
            xmax=max(0, min(self.xmax, width - 1)),
            ymax=max(0, min(self.ymax, height - 1)),
        )

    def to_yolo(self, width: int, height: int) -> Tuple[float, float, float, float]:
        x_center = (self.xmin + self.xmax) / 2.0 / width
        y_center = (self.ymin + self.ymax) / 2.0 / height
        box_width = (self.xmax - self.xmin) / width
        box_height = (self.ymax - self.ymin) / height
        return x_center, y_center, box_width, box_height


@dataclass
class LabelConfig:
    """Contour filters used when converting flow masks to labels."""

    debris_threshold: int = 10
    min_width: int = 50
    min_height: int = 50
    margin: int = 50
    min_area: int = 80 * 80
    class_id: int = 0


def index_images(image_dir: Path) -> Dict[str, Path]:
    """Index images by both filename and stem for robust frame matching."""

    indexed: Dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            indexed[path.name] = path
            indexed[path.stem] = path
    return indexed


def extract_motion_box(mask_image: np.ndarray, config: Optional[LabelConfig] = None) -> Optional[Box]:
    """Extract one bounding box around valid moving sonar debris regions."""

    config = config or LabelConfig()
    gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) if mask_image.ndim == 3 else mask_image
    _, binary = cv2.threshold(gray, config.debris_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    xs1: List[int] = []
    ys1: List[int] = []
    xs2: List[int] = []
    ys2: List[int] = []
    height, width = gray.shape

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        touches_border = x == 0 and y == 0 and w >= width - 1 and h >= height - 1
        too_small = (
            w < config.min_width
            or h < config.min_height
            or x < config.margin
            or y < config.margin
            or area < config.min_area
        )

        if touches_border or too_small:
            continue

        xs1.append(x)
        ys1.append(y)
        xs2.append(x + w)
        ys2.append(y + h)

    if not xs1:
        return None

    return Box(min(xs1), min(ys1), max(xs2), max(ys2)).clamp(width, height)


def write_yolo_label(label_path: Path, box: Box, width: int, height: int, class_id: int) -> None:
    """Write a single YOLO-format label file."""

    x_center, y_center, box_width, box_height = box.to_yolo(width, height)
    with label_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}\n"
        )


def generate_pseudo_labels(
    flow_images_dir: Path,
    sonar_images_dir: Path,
    output_dir: Path,
    csv_path: Path,
    pattern: str = "*",
    config: Optional[LabelConfig] = None,
) -> List[Dict[str, object]]:
    """Generate YOLO labels and visualization images from flow masks."""

    config = config or LabelConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_images = list_images(flow_images_dir, pattern)
    sonar_index = index_images(sonar_images_dir)

    rows: List[Dict[str, object]] = []
    last_box: Optional[Box] = None

    for flow_path in flow_images:
        sonar_path = sonar_index.get(flow_path.name) or sonar_index.get(flow_path.stem)
        if sonar_path is None:
            raise FileNotFoundError(
                f"No matching sonar frame for {flow_path.name} in {sonar_images_dir}"
            )

        flow_image = read_image(flow_path, grayscale=False)
        sonar_image = read_image(sonar_path, grayscale=False)
        if sonar_image.shape[:2] != flow_image.shape[:2]:
            raise ValueError(
                f"Image size mismatch for {flow_path.name}: "
                f"flow={flow_image.shape[:2]}, sonar={sonar_image.shape[:2]}"
            )
        height, width = flow_image.shape[:2]

        box = extract_motion_box(flow_image, config)
        if box is None:
            box = last_box
        else:
            last_box = box

        if box is None:
            continue

        box = box.clamp(width, height)
        label_path = output_dir / f"{flow_path.stem}.txt"
        write_yolo_label(label_path, box, width, height, config.class_id)

        cv2.rectangle(
            sonar_image,
            (box.xmin, box.ymin),
            (box.xmax, box.ymax),
            (0, 255, 0),
            4,
        )
        visualization_name = f"{flow_path.stem}.jpg"
        cv2.imwrite(str(output_dir / visualization_name), sonar_image)

        rows.append(
            {
                "img_name": visualization_name,
                "label": config.class_id,
                "xmin": box.xmin / width,
                "ymin": box.ymin / height,
                "xmax": box.xmax / width,
                "ymax": box.ymax / height,
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["img_name", "label", "xmin", "ymin", "xmax", "ymax"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows
