"""Calcola feature manuali interpretabili a partire dai crop cellulari."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nmcd_common import DEFAULT_OUTPUT_ROOT, ensure_dir


METADATA_COLUMNS = {
    "split",
    "annotation_id",
    "image_id",
    "source_file",
    "category_id",
    "label",
    "crop_file",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract handcrafted features from cell crops.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "metadata" / "cell_crops.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "features" / "handcrafted_features.csv",
    )
    return parser.parse_args()


def read_image_rgb(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def nucleus_like_mask(rgb: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    if fg_mask.sum() == 0:
        return np.zeros(fg_mask.shape, dtype=bool)

    fg_gray = gray[fg_mask]
    dark_cutoff = np.percentile(fg_gray, 35)
    mask = fg_mask & (gray <= dark_cutoff) & (saturation >= 25) & (value <= 220)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    return mask_u8 > 0


def channel_stats(prefix: str, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        pixels = image.reshape(-1, image.shape[-1])
    else:
        pixels = image[mask]
        if len(pixels) == 0:
            pixels = image.reshape(-1, image.shape[-1])

    features: dict[str, float] = {}
    for idx in range(image.shape[-1]):
        values = pixels[:, idx].astype(np.float32)
        features[f"{prefix}_ch{idx}_mean"] = float(np.mean(values))
        features[f"{prefix}_ch{idx}_std"] = float(np.std(values))
        features[f"{prefix}_ch{idx}_p10"] = float(np.percentile(values, 10))
        features[f"{prefix}_ch{idx}_p50"] = float(np.percentile(values, 50))
        features[f"{prefix}_ch{idx}_p90"] = float(np.percentile(values, 90))
    return features


def shape_features(mask: np.ndarray) -> dict[str, float]:
    height, width = mask.shape
    total_area = float(height * width)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "fg_area": 0.0,
            "fg_area_ratio": 0.0,
            "fg_perimeter": 0.0,
            "fg_circularity": 0.0,
            "fg_solidity": 0.0,
            "fg_extent": 0.0,
            "fg_aspect_ratio": float(width / max(height, 1)),
        }

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    circularity = 0.0
    if perimeter > 0:
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))

    return {
        "fg_area": area,
        "fg_area_ratio": float(area / total_area) if total_area else 0.0,
        "fg_perimeter": perimeter,
        "fg_circularity": circularity,
        "fg_solidity": float(area / hull_area) if hull_area > 0 else 0.0,
        "fg_extent": float(area / (bbox_w * bbox_h)) if bbox_w * bbox_h > 0 else 0.0,
        "fg_aspect_ratio": float(bbox_w / max(bbox_h, 1)),
    }


def count_components(mask: np.ndarray, min_area: float, max_area: float) -> int:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    count = 0
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            count += 1
    return count


def texture_features(gray: np.ndarray) -> dict[str, float]:
    values = gray.astype(np.float32)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=False)
    probabilities = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    quantized = np.floor(gray.astype(np.float32) / 16).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )

    features = {
        "gray_mean": float(np.mean(values)),
        "gray_std": float(np.std(values)),
        "gray_entropy": entropy,
    }
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        features[f"glcm_{prop}"] = float(np.mean(graycoprops(glcm, prop)))
    return features


def extract_features(crop_path: Path) -> dict[str, float]:
    rgb = read_image_rgb(crop_path)
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    fg = foreground_mask(rgb)
    fg_area = float(fg.sum())
    nucleus_mask = nucleus_like_mask(rgb, fg)
    nucleus_area = float(nucleus_mask.sum())

    features: dict[str, float] = {
        "crop_width": float(width),
        "crop_height": float(height),
        "crop_area": float(width * height),
        "crop_aspect_ratio": float(width / max(height, 1)),
        "nucleus_like_area_ratio": float(nucleus_area / max(fg_area, 1.0)),
        "nucleus_like_count": float(
            count_components(
                nucleus_mask,
                min_area=max(5.0, fg_area * 0.003),
                max_area=max(20.0, fg_area * 0.45),
            )
        ),
    }
    features.update(shape_features(fg))
    features.update(channel_stats("rgb_all", rgb))
    features.update(channel_stats("rgb_fg", rgb, fg))
    features.update(channel_stats("hsv_fg", hsv, fg))
    features.update(channel_stats("lab_fg", lab, fg))
    features.update(texture_features(gray))
    return features


def main() -> None:
    args = parse_args()
    if not args.metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata not found: {args.metadata_csv}. Run 02_export_cell_crops.py first."
        )

    metadata = pd.read_csv(args.metadata_csv)
    rows = []
    for record in tqdm(metadata.to_dict("records"), desc="Extracting features"):
        crop_path = ROOT / record["crop_file"]
        feature_row = {key: record[key] for key in record if key in METADATA_COLUMNS}
        feature_row.update(extract_features(crop_path))
        rows.append(feature_row)

    ensure_dir(args.output_csv.parent)
    features = pd.DataFrame(rows)
    features.to_csv(args.output_csv, index=False)
    print(f"Wrote features to: {args.output_csv}")
    print(f"Rows: {len(features)}")
    print(f"Feature columns: {len([c for c in features.columns if c not in METADATA_COLUMNS])}")


if __name__ == "__main__":
    main()
