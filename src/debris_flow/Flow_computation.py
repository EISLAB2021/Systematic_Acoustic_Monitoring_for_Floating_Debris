"""Dense optical-flow utilities for sonar debris tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class FarnebackConfig:
    """Parameters used by Farneback(FB) dense optical flow."""

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 5
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    threshold_std_factor: float = 0.5
    enhance_brightness: bool = True
    direction_correction: bool = False
    region_expansion: bool = False
    expansion_radius: int = 4
    angle_threshold: float = np.pi / 4


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from disk and fail with a clear message when missing."""

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Improve local contrast with CLAHE."""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(image).astype(np.uint8)


def flow_threshold(flow: np.ndarray, std_factor: float = 0.5) -> float:
    """Set a motion threshold from the flow magnitude distribution."""

    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude) + std_factor * np.std(magnitude))


def compute_dense_flow(
    frame1_gray: np.ndarray,
    frame2_gray: np.ndarray,
    config: Optional[FarnebackConfig] = None,
) -> Tuple[np.ndarray, float]:
    """Compute dense optical flow between two grayscale sonar frames."""

    config = config or FarnebackConfig()
    if frame1_gray.shape != frame2_gray.shape:
        raise ValueError("Input sonar frames must have the same dimensions.")

    if config.enhance_brightness:
        frame1_gray = enhance_contrast(frame1_gray)
        frame2_gray = enhance_contrast(frame2_gray)

    start = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray,
        frame2_gray,
        None,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    threshold = flow_threshold(flow, config.threshold_std_factor)
    valid_mask = np.hypot(flow[..., 0], flow[..., 1]) > threshold

    if config.direction_correction:
        flow = correct_flow_direction(flow, valid_mask, config.angle_threshold)
    if config.region_expansion:
        flow = expand_flow_regions(flow, valid_mask, config.expansion_radius)

    return flow, elapsed_ms


def correct_flow_direction(
    flow: np.ndarray,
    valid_mask: np.ndarray,
    angle_threshold: float,
) -> np.ndarray:
    """Smooth local flow directions with neighboring valid vectors."""

    corrected = flow.copy()
    height, width = valid_mask.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if not valid_mask[y, x]:
                continue

            neighbor_flows = flow[y - 1 : y + 2, x - 1 : x + 2]
            neighbor_valid = valid_mask[y - 1 : y + 2, x - 1 : x + 2]
            neighbor_dirs = np.arctan2(
                neighbor_flows[..., 1], neighbor_flows[..., 0]
            )[neighbor_valid]

            if neighbor_dirs.size == 0:
                continue

            current_dir = np.arctan2(flow[y, x, 1], flow[y, x, 0])
            dir_diff = np.abs(neighbor_dirs - current_dir)
            dir_diff = np.minimum(dir_diff, 2 * np.pi - dir_diff)
            similar_dirs = neighbor_dirs[dir_diff < angle_threshold]

            if similar_dirs.size == 0:
                continue

            mean_dir = np.arctan2(
                np.mean(np.sin(similar_dirs)),
                np.mean(np.cos(similar_dirs)),
            )
            speed = np.hypot(flow[y, x, 0], flow[y, x, 1])
            corrected[y, x, 0] = speed * np.cos(mean_dir)
            corrected[y, x, 1] = speed * np.sin(mean_dir)

    return corrected


def expand_flow_regions(
    flow: np.ndarray,
    valid_mask: np.ndarray,
    radius: int = 4,
) -> np.ndarray:
    """Fill small invalid holes with the mean nearby valid flow vector."""

    expanded = flow.copy()
    height, width = valid_mask.shape

    for y in range(height):
        for x in range(width):
            if valid_mask[y, x]:
                continue

            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            neighbor_valid = valid_mask[y0:y1, x0:x1]
            if not np.any(neighbor_valid):
                continue

            valid_vectors = flow[y0:y1, x0:x1][neighbor_valid]
            expanded[y, x] = np.mean(valid_vectors, axis=0)

    return expanded


def flow_to_bgr(flow: np.ndarray, magnitude_gain: float = 2.0) -> np.ndarray:
    """Convert a dense flow field to a BGR color visualization."""

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)
    hsv[..., 1] = 255
    value = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = np.uint8(np.clip(value * magnitude_gain, 0, 255))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(
    image: np.ndarray,
    flow: np.ndarray,
    threshold: Optional[float] = None,
    step: int = 4,
    scale: float = 2.0,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw sparse motion vectors over an image."""

    threshold = flow_threshold(flow) if threshold is None else threshold
    arrows = image.copy()

    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            dx, dy = flow[y, x]
            if np.hypot(dx, dy) <= threshold:
                continue

            end_point = (int(x + dx * scale), int(y + dy * scale))
            cv2.arrowedLine(arrows, (x, y), end_point, color, 1, tipLength=0.3)

    return arrows


def retain_vertical_region(
    image: np.ndarray,
    top_fraction: float = 0.0,
    bottom_fraction: float = 0.0,
    draw_roi: bool = False,
) -> np.ndarray:
    """Keep a vertical region of interest and set the rest to black."""

    if not 0 <= top_fraction < 1 or not 0 <= bottom_fraction < 1:
        raise ValueError("ROI fractions must be in the range [0, 1).")

    height = image.shape[0]
    start = int(height * top_fraction)
    end = height - int(height * bottom_fraction)
    if start >= end:
        raise ValueError("The selected ROI is empty.")

    masked = np.zeros_like(image)
    masked[start:end, ...] = image[start:end, ...]

    if draw_roi:
        cv2.rectangle(masked, (1, start), (image.shape[1] - 2, end - 1), (0, 0, 255), 1)

    return masked


def find_motion_boxes(
    mask_image: np.ndarray,
    debris_threshold: int = 10,
    min_area: int = 200,
    min_width: int = 7,
) -> List[Tuple[int, int, int, int, int]]:
    """Find connected motion regions in a flow visualization or binary mask."""

    if mask_image.ndim == 3:
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_image

    _, binary = cv2.threshold(gray, debris_threshold, 255, cv2.THRESH_BINARY)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes: List[Tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= min_area and w >= min_width:
            boxes.append((int(x), int(y), int(w), int(h), int(area)))

    return boxes


def annotate_motion_regions(
    mask_image: np.ndarray,
    reference_image: np.ndarray,
    output_dir: Path,
    debris_threshold: int = 10,
    min_area: int = 200,
    min_width: int = 7,
    prefix: str = "motion_regions",
) -> List[Tuple[int, int, int, int, int]]:
    """Draw motion-region boxes on a reference frame and save a text summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    boxes = find_motion_boxes(mask_image, debris_threshold, min_area, min_width)
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] // 2, reverse=True)

    annotated = reference_image.copy()
    if reference_image.ndim == 3:
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference_image

    if mask_image.ndim == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image
    _, binary = cv2.threshold(mask_gray, debris_threshold, 255, cv2.THRESH_BINARY)

    text_path = output_dir / f"{prefix}_boxes.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        handle.write("index,center_x,center_y,pixel_count,average_intensity\n")
        for index, (x, y, w, h, area) in enumerate(boxes, start=1):
            x2, y2 = x + w, y + h
            center_x, center_y = x + w // 2, y + h // 2
            box_mask = binary[y:y2, x:x2] == 255
            pixels = reference_gray[y:y2, x:x2][box_mask]
            average = float(np.mean(pixels)) if pixels.size else 0.0

            cv2.rectangle(annotated, (x, y), (x2, y2), (0, 255, 255), 2)
            cv2.drawMarker(
                annotated,
                (center_x, center_y),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
            )
            cv2.putText(
                annotated,
                str(index),
                (center_x - 5, min(y2 + 25, annotated.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 228, 0),
                2,
            )
            handle.write(f"{index},{center_x},{center_y},{area},{average:.2f}\n")

        handle.write(f"total,{len(boxes)}\n")
        handle.write(f"debris_threshold,{debris_threshold}\n")
        handle.write(f"min_area,{min_area}\n")

    cv2.imwrite(str(output_dir / f"{prefix}.png"), annotated)
    return boxes


def run_pair(
    frame1_path: Path,
    frame2_path: Path,
    output_dir: Path,
    config: Optional[FarnebackConfig] = None,
    roi_top: float = 0.0,
    roi_bottom: float = 0.0,
    annotate_boxes: bool = False,
    box_min_area: int = 200,
) -> Dict[str, object]:
    """Run the two-frame optical-flow workflow and save visual outputs."""

    config = config or FarnebackConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame1 = read_image(frame1_path, grayscale=False)
    frame2 = read_image(frame2_path, grayscale=False)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow, elapsed_ms = compute_dense_flow(gray1, gray2, config)
    threshold = flow_threshold(flow, config.threshold_std_factor)

    flow_image = retain_vertical_region(
        flow_to_bgr(flow),
        top_fraction=roi_top,
        bottom_fraction=roi_bottom,
        draw_roi=False,
    )
    arrows = retain_vertical_region(
        draw_flow_arrows(frame1, flow, threshold=threshold),
        top_fraction=roi_top,
        bottom_fraction=roi_bottom,
        draw_roi=True,
    )

    flow_path = output_dir / "optical_flow_dense.png"
    arrows_path = output_dir / "motion_vectors_dense.png"
    cv2.imwrite(str(flow_path), flow_image)
    cv2.imwrite(str(arrows_path), arrows)

    boxes: Sequence[Tuple[int, int, int, int, int]] = []
    if annotate_boxes:
        boxes = annotate_motion_regions(
            flow_image,
            frame2,
            output_dir / "boxes",
            min_area=box_min_area,
        )

    return {
        "elapsed_ms": elapsed_ms,
        "threshold": threshold,
        "flow_image": flow_path,
        "arrow_image": arrows_path,
        "boxes": list(boxes),
    }


def list_images(input_dir: Path, pattern: str = "*") -> List[Path]:
    """Return sorted image files from a directory."""

    files = sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir} with pattern {pattern}")
    return files


def filter_dominant_downward_motion(
    flow: np.ndarray,
    down_threshold: float = 1.0,
    dbscan_eps: float = 25.0,
    dbscan_min_samples: int = 2,
) -> np.ndarray:
    """Keep the dominant cluster of downward flow vectors."""

    from sklearn.cluster import DBSCAN

    downward_mask = flow[..., 1] > down_threshold
    ys, xs = np.nonzero(downward_mask)
    filtered = np.zeros_like(flow)

    if xs.size == 0:
        return filtered

    vectors = flow[ys, xs]
    if xs.size < dbscan_min_samples:
        filtered[ys, xs] = vectors
        return filtered

    data = np.column_stack((xs, ys, vectors))
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(data).labels_
    non_noise = labels[labels != -1]

    if non_noise.size:
        unique, counts = np.unique(non_noise, return_counts=True)
        main_label = unique[np.argmax(counts)]
        keep = labels == main_label
    else:
        keep = np.ones_like(labels, dtype=bool)

    filtered[ys[keep], xs[keep]] = vectors[keep]
    return filtered


def build_flow_image_sequence(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*",
    config: Optional[FarnebackConfig] = None,
    down_threshold: float = 1.0,
    dbscan_eps: float = 25.0,
    dbscan_min_samples: int = 2,
) -> List[Path]:
    """Generate filtered optical-flow visualizations for an image sequence."""

    config = config or FarnebackConfig(winsize=12)
    files = list_images(input_dir, pattern)
    if len(files) < 2:
        raise ValueError("At least two SONAR frames are required for optical flow.")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    previous = read_image(files[0], grayscale=True)

    for index in range(1, len(files)):
        current = read_image(files[index], grayscale=True)
        flow, _ = compute_dense_flow(previous, current, config)
        filtered = filter_dominant_downward_motion(
            flow,
            down_threshold=down_threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )

        output_path = output_dir / files[index - 1].name
        cv2.imwrite(str(output_path), flow_to_bgr(filtered, magnitude_gain=1.0))
        saved_paths.append(output_path)
        previous = current

    return saved_paths
