"""Projection helpers for converting sonar frames to fan-shaped views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class ProjectionConfig:
    """Geometry parameters for a rectangular-to-fan sonar projection."""

    start_row: int = 400
    end_row: int = 1100
    range_min: float = 0.971
    range_resolution: float = 0.0029
    beam_width_deg: float = 30.0
    scaled_width: int = 128
    rotation_deg: float = 15.0
    sector_edge_deg: float = 75.0


def read_grayscale(path: Path) -> np.ndarray:
    """Load an image as an 8-bit grayscale array."""

    return np.array(Image.open(path).convert("L"))


def project_to_fan(image: np.ndarray, config: Optional[ProjectionConfig] = None) -> np.ndarray:
    """Project a rectangular sonar image into a fan-shaped coordinate view."""

    config = config or ProjectionConfig()
    start_row = max(0, min(config.start_row, image.shape[0]))
    end_row = max(start_row + 1, min(config.end_row, image.shape[0]))
    cropped = image[start_row:end_row]

    if cropped.shape[1] != config.scaled_width:
        cropped = cv2.resize(
            cropped,
            (config.scaled_width, cropped.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    upper_range = config.range_min + end_row * config.range_resolution
    lower_range = config.range_min + start_row * config.range_resolution
    radial_length = int((upper_range - lower_range) / config.range_resolution)
    full_length = int(upper_range / config.range_resolution)
    fan_width = int(
        upper_range
        / config.range_resolution
        * math.sin(math.radians(config.beam_width_deg))
    )

    raw_fan = np.zeros((full_length, fan_width), dtype=np.float32)
    for row in range(full_length):
        for col in range(fan_width):
            radius = math.hypot(row, col)
            if radius <= 0:
                continue

            radial_index = int(radius - lower_range / config.range_resolution)
            if radial_index <= 0 or radial_index >= radial_length:
                continue

            angle_index = int(
                config.scaled_width
                * 180
                / config.beam_width_deg
                / math.pi
                * math.acos(row / radius)
            )
            if 0 <= angle_index < config.scaled_width:
                raw_fan[row, col] = cropped[radial_index, angle_index]

    rotated_width = int(full_length * math.cos(math.radians(config.sector_edge_deg)) * 2)
    offset = int(full_length - full_length * math.sin(math.radians(config.sector_edge_deg)))
    rotated_height = full_length + offset
    y_center = full_length

    rotated = np.zeros((rotated_height, rotated_width), dtype=np.float32)
    cos_theta = math.cos(math.radians(config.rotation_deg))
    sin_theta = math.sin(math.radians(config.rotation_deg))

    for row in range(rotated_height):
        for col in range(rotated_width):
            px = int(col * cos_theta + (row - y_center) * sin_theta)
            py = int(-col * sin_theta + (row - y_center) * cos_theta + y_center)

            if px < 0 or px >= fan_width or py < 0 or py >= full_length:
                continue
            if row - offset > 0:
                rotated[row - offset, col] = raw_fan[py, px]

    fan = rotated[: full_length - 1, :]
    fan = cv2.flip(fan, 0)
    return np.clip(fan, 0, 255).astype(np.uint8)


def save_fan_projection(
    input_path: Path,
    output_path: Path,
    config: Optional[ProjectionConfig] = None,
) -> Path:
    """Project one sonar frame and save the fan-shaped result."""

    image = read_grayscale(input_path)
    fan = project_to_fan(image, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), fan)
    return output_path
